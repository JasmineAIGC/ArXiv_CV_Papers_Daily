"""ArXiv CV 论文每日获取：主入口

流程：抓取 ArXiv cs.CV 论文 → 并行翻译标题/分析贡献/关键词分类 → 生成 Markdown 报告
"""
import os
import re
import time
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import arxiv
from tqdm import tqdm

import feedparser
import requests

from categories_config import CATEGORY_KEYWORDS
from classifier import get_category_by_keywords
from llm_helper import LLMHelper
from markdown_output import save_papers_to_markdown

# 查询参数集中维护在 config.py，这里统一导入
from config import (
    ARXIV_QUERY, ARXIV_PAGE_SIZE, ARXIV_DELAY_SECONDS,
    ARXIV_NUM_RETRIES, ARXIV_BATCH_SIZE,
    QUERY_DAYS_AGO, MAX_RESULTS, MAX_WORKERS,
    LLM_MAX_WORKERS, MAX_AUTHORS_SHOWN,
    ENABLE_TITLE_TRANSLATION, ENABLE_CONTRIBUTION_ANALYSIS,
    ENABLE_LLM_ARBITRATION,
)

# 兼容 arxiv 1.4.8 的 HTTP 重定向行为，强制使用 HTTPS 查询端点
ARXIV_API_URL_FORMAT = "https://export.arxiv.org/api/query?{}"

# ---------------------------------------------------------------------------
# 修复 arXiv API HTTP 406 问题：
# arxiv 库底层通过 feedparser → urllib 发请求，urllib 的请求特征
# （HTTP/1.1 + Connection: close + 无 Accept 头）会被 export.arxiv.org
# 的反爬层拒绝，返回 HTTP 406 Not Acceptable。
# 这里拦截发往 arXiv 的 feed 请求，改用 requests（默认 keep-alive + 标准头）
# 抓取后再交给 feedparser 解析。
# ---------------------------------------------------------------------------
_ARXIV_FEED_URL_PREFIX = "https://export.arxiv.org/api/query"
_ORIGINAL_FEEDPARSER_PARSE = feedparser.parse


def _parse_arxiv_feed(url, *args, **kwargs):
    if isinstance(url, str) and url.startswith(_ARXIV_FEED_URL_PREFIX):
        resp = requests.get(
            url,
            headers={"User-Agent": "ArXivCV-Daily/1.0"},
            timeout=60,
        )
        feed = _ORIGINAL_FEEDPARSER_PARSE(resp.content, *args, **kwargs)
        feed["status"] = resp.status_code
        return feed
    return _ORIGINAL_FEEDPARSER_PARSE(url, *args, **kwargs)


feedparser.parse = _parse_arxiv_feed


def extract_github_link(text: str):
    """从文本中提取GitHub链接

    Args:
        text: 论文摘要文本

    Returns:
        str: GitHub链接或None
    """
    # GitHub链接模式（从具体到宽松排列）
    github_patterns = [
        # 带协议或 www 前缀的仓库链接
        r'https?://(?:www\.)?github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        # 无协议的仓库链接
        r'(?:www\.)?github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9-_.]+',
        # 项目主页
        r'https?://[a-zA-Z0-9-]+\.github\.io/[a-zA-Z0-9-_.]+',
    ]

    # 从摘要中查找，返回第一个匹配
    for pattern in github_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            url = match.group(0)
            if not url.startswith('http'):
                url = 'https://' + url
            return url

    return None


def process_paper(paper, llm_helper, target_date, llm_executor=None):
    """处理单篇论文的所有分析任务

    Args:
        paper: ArXiv论文对象
        llm_helper: LLM助手实例
        target_date: 目标日期
        llm_executor: 共享的LLM任务线程池（可选，用于并发执行翻译和分析）

    Returns:
        Dict: 包含论文信息的字典，如果论文不符合日期要求则返回None
    """
    try:
        # 获取论文信息
        title = paper.title
        abstract = paper.summary
        paper_url = paper.entry_id
        author_list = paper.authors
        authors = [author.name for author in author_list]
        authors_str = ', '.join(authors[:MAX_AUTHORS_SHOWN]) + (' .etc.' if len(authors) > MAX_AUTHORS_SHOWN else '')
        published = paper.published
        updated = paper.updated

        # 检查发布日期或更新日期是否匹配目标日期
        published_date = published.date()
        updated_date = updated.date()
        if published_date != target_date and updated_date != target_date:
            return None

        # 获取PDF链接
        pdf_url = next(
            (link.href for link in paper.links if link.title == "pdf"), None)

        # 初始化默认值，避免异常时未定义
        github_link = extract_github_link(abstract)
        category = "其他 (Others)"
        subcategory = "未指定"
        title_cn = f"[翻译失败] {title}"
        analysis = {}

        # 执行耗时任务（标题翻译 + 贡献分析），使用共享线程池控制并发
        # 受 config.py 中 ENABLE_TITLE_TRANSLATION / ENABLE_CONTRIBUTION_ANALYSIS 控制
        try:
            if ENABLE_CONTRIBUTION_ANALYSIS and ENABLE_TITLE_TRANSLATION:
                if llm_executor is not None:
                    analysis_future = llm_executor.submit(
                        llm_helper.analyze_paper_contribution, title, abstract)
                    title_cn_future = llm_executor.submit(
                        llm_helper.translate_title, title)
                    analysis = analysis_future.result() or {}
                    title_cn = title_cn_future.result() or f"[翻译失败] {title}"
                else:
                    analysis = llm_helper.analyze_paper_contribution(title, abstract) or {}
                    title_cn = llm_helper.translate_title(title) or f"[翻译失败] {title}"
            else:
                if ENABLE_CONTRIBUTION_ANALYSIS:
                    analysis = llm_helper.analyze_paper_contribution(title, abstract) or {}
                if ENABLE_TITLE_TRANSLATION:
                    title_cn = llm_helper.translate_title(title) or f"[翻译失败] {title}"
                else:
                    title_cn = title  # 关闭翻译时使用英文标题
        except Exception as e:
            print(f"并行处理任务时出错: {str(e)}")
            # 继续处理，使用默认值

        # 使用基于关键词的分类方法
        try:
            category_results = get_category_by_keywords(title, abstract, CATEGORY_KEYWORDS)

            if category_results:
                # 获取主类别和得分
                result_item = category_results[0]

                # 兼容多种返回格式：(category, score) 或 (category, score, subcategory) 或 (category, score, subcategory, explanation)
                if len(result_item) >= 4:  # 新格式，包含解释
                    main_category, _, sub_category_tuple, _ = result_item
                elif len(result_item) == 3:  # 旧格式，包含子类别
                    main_category, _, sub_category_tuple = result_item
                else:  # 最简单的格式
                    main_category, _ = result_item
                    sub_category_tuple = None

                category = main_category

                # 处理子类别
                if sub_category_tuple:
                    subcategory_name, subcategory_score = sub_category_tuple
                    subcategory = subcategory_name
                else:
                    subcategory = "未指定"
            else:
                # 如果没有匹配的类别，使用默认类别
                category = "其他 (Others)"
                subcategory = "未指定"
        except Exception as e:
            print(f"分类论文时出错: {str(e)}")
            traceback.print_exc()
            category = "其他 (Others)"
            subcategory = "未指定"

        paper_info = {
            'title': title,
            'title_zh': title_cn,
            'abstract': abstract,
            'authors': authors_str,
            'pdf_url': pdf_url,
            'github_url': github_link,
            'url': paper_url,
            'category': category,
            'subcategory': subcategory,
            'published': published,
            'updated': updated,
            'is_updated': updated_date == target_date and published_date != target_date
        }

        # 合并分析结果
        if analysis:
            paper_info.update(analysis)

        return paper_info

    except Exception as e:
        print(f"处理论文时出错: {str(e)}")
        return None


def get_cv_papers():
    """获取CV领域论文并保存为Markdown"""
    print("\n" + "=" * 50)
    print(f"开始获取CV论文 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    try:
        # 获取目标日期（前一天）
        target_date = (datetime.now() - timedelta(days=QUERY_DAYS_AGO)).date()
        print(f"\n📅 目标日期: {target_date}")
        print(f"📊 最大论文数: {MAX_RESULTS}")
        print(f"🧵 论文处理线程数: {MAX_WORKERS}")
        print(f"🤖 LLM任务线程数: {LLM_MAX_WORKERS}")
        print(f"🌐 标题翻译: {'开' if ENABLE_TITLE_TRANSLATION else '关'} | 贡献分析: {'开' if ENABLE_CONTRIBUTION_ANALYSIS else '关'} | LLM仲裁: {'开' if ENABLE_LLM_ARBITRATION else '关'}\n")

        # 初始化LLM助手
        print("🤖 初始化LLM助手...")
        llm_helper = LLMHelper()

        # 初始化arxiv客户端（参数来自 config.py）
        print("🔄 初始化arXiv客户端...")
        client = arxiv.Client(
            page_size=ARXIV_PAGE_SIZE,
            delay_seconds=ARXIV_DELAY_SECONDS,
            num_retries=ARXIV_NUM_RETRIES
        )
        client.query_url_format = ARXIV_API_URL_FORMAT

        # 构建查询
        search = arxiv.Search(
            query=ARXIV_QUERY,
            max_results=MAX_RESULTS,
            sort_by=arxiv.SortCriterion.LastUpdatedDate,
            sort_order=arxiv.SortOrder.Descending  # 确保按时间降序排序
        )

        # 创建线程池
        total_papers = 0
        papers_by_category = defaultdict(list)

        # 确保"其他 (Others)"类别总是存在
        papers_by_category["其他 (Others)"]  # 初始化空列表

        # 外层线程池处理论文，内层共享线程池执行LLM任务，
        # 避免每篇论文反复创建线程池，同时控制API并发
        with ThreadPoolExecutor(max_workers=LLM_MAX_WORKERS) as llm_executor, \
                ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 创建进度条
            print("\n🔍 开始获取论文...")
            try:
                results = client.results(search)
            except Exception as e:
                print(f"⚠️ arXiv 首次请求失败，等待 10 秒后重试: {e}")
                time.sleep(10)
                results = client.results(search)

            # 创建总进度条
            total_pbar = tqdm(
                total=MAX_RESULTS,
                desc="总进度",
                unit="篇",
                position=0,
                leave=True
            )

            # 创建批处理进度条
            batch_pbar = tqdm(
                total=0,  # 初始值为0，后面会更新
                desc="当前批次",
                unit="篇",
                position=1,
                leave=True
            )

            # 批量处理论文
            batch_size = ARXIV_BATCH_SIZE
            papers = []

            for i, paper in enumerate(results):
                papers.append(paper)

                # 当收集到一批论文或达到最大数量时处理
                if len(papers) >= batch_size or i >= MAX_RESULTS - 1:
                    batch_pbar.reset()  # 重置批处理进度条
                    batch_pbar.total = len(papers)  # 设置正确的总数

                    # 提交所有任务
                    batch_futures = [
                        executor.submit(process_paper, paper, llm_helper, target_date, llm_executor)
                        for paper in papers
                    ]

                    # 等待当前批次完成
                    for future in as_completed(batch_futures):
                        paper_info = future.result()
                        if paper_info:  # 如果论文符合日期要求
                            total_papers += 1
                            category = paper_info['category']
                            papers_by_category[category].append(paper_info)
                            total_pbar.update(1)  # 更新总进度
                        batch_pbar.update(1)  # 更新批处理进度

                    # 清空当前批次
                    papers = []

                # 如果达到最大数量，停止获取
                if i >= MAX_RESULTS - 1:
                    break

            # 处理剩余的论文
            if papers:
                batch_pbar.reset()  # 重置批处理进度条
                batch_pbar.total = len(papers)  # 设置正确的总数

                # 提交所有任务
                batch_futures = [
                    executor.submit(process_paper, paper, llm_helper, target_date, llm_executor)
                    for paper in papers
                ]

                # 等待所有任务完成
                for future in as_completed(batch_futures):
                    paper_info = future.result()
                    if paper_info:  # 如果论文符合日期要求
                        total_papers += 1
                        category = paper_info['category']
                        papers_by_category[category].append(paper_info)
                        total_pbar.update(1)  # 更新总进度
                    batch_pbar.update(1)  # 更新批处理进度

            # 关闭进度条
            batch_pbar.close()
            total_pbar.close()

        if total_papers == 0:
            print(f"没有找到{target_date}发布的论文。")
            return

        # 打印统计信息
        print(f"\n📊 论文统计信息：")
        print(f"{'=' * 50}")

        # 按论文数量降序排序类别
        sorted_categories = sorted(
            papers_by_category.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # 统计总论文数
        total_papers = sum(len(papers) for papers in papers_by_category.values())

        # 一级分类和对应的二级分类统计
        for category, papers in sorted_categories:
            # 即使没有论文，也要打印"其他"类别
            if len(papers) == 0 and category != "其他 (Others)":
                continue

            # 打印一级分类标题
            print(f"\n【{category}】")

            # 如果不是"其他"类别，将没有子类别的论文移动到"其他"类别
            if category != "其他 (Others)":
                # 分离有子类别的论文和无子类别的论文
                for paper in list(papers):  # 创建副本以避免在遍历时修改
                    subcategory = paper.get('subcategory', '')
                    if not subcategory or subcategory == "未指定":
                        # 将没有子类别的论文移动到"其他"类别
                        papers_by_category["其他 (Others)"].append(paper)
                        papers.remove(paper)  # 从当前类别中移除

            # 如果当前类别下没有论文，跳过
            if len(papers) == 0 and category != "其他 (Others)":
                continue

            # 按子类别分组论文
            papers_by_subcategory = defaultdict(list)
            for paper in papers:  # 使用更新后的papers
                subcategory = paper.get('subcategory', '')
                if subcategory and subcategory != "未指定":
                    papers_by_subcategory[subcategory].append(paper)
                else:
                    # 对于没有子类别的论文，如果是"其他"类别，显示为"未分类"
                    papers_by_subcategory["未分类"].append(paper)

            # 按论文数量降序排序子类别
            sorted_subcategories = sorted(
                papers_by_subcategory.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )

            # 打印一级分类总数
            num_new = sum(1 for p in papers if not p['is_updated'])
            num_updated = sum(1 for p in papers if p['is_updated'])
            print(f"总计: {len(papers):3d} 篇 (🆕 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")

            # 打印子类别统计
            for subcategory, subpapers in sorted_subcategories:
                num_new = sum(1 for p in subpapers if not p['is_updated'])
                num_updated = sum(1 for p in subpapers if p['is_updated'])
                print(
                    f"└─ {subcategory:15s}: {len(subpapers):3d} 篇 (🆕 {num_new:3d} 新发布, 📝 {num_updated:3d} 更新)")

        print(f"\n{'=' * 50}")
        print(f"总计: {total_papers} 篇")

        # 保存结果到Markdown文件
        print("\n💾 正在保存结果到Markdown文件...")
        save_papers_to_markdown(papers_by_category, target_date)

        print("\n" + "=" * 50)
        print(f"CV论文获取完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 50 + "\n")

    except Exception as e:
        print("\n❌ 处理CV论文时出错:")
        print(f"错误信息: {str(e)}")
        print(f"发生时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        raise  # 抛出异常以便查看详细错误信息


if __name__ == "__main__":
    # 直接运行查询
    get_cv_papers()
