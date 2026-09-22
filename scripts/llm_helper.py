"""LLM助手：论文标题翻译、核心贡献分析与分类仲裁（支持豆包、GLM（智谱）和 DeepSeek）"""
import time
from typing import Tuple, List, Dict

# 导入配置
from config import (
    LLM_PROVIDER,
    LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_RETRY_DELAY,
    TRANSLATE_TEMPERATURE, TRANSLATE_MAX_TOKENS, TRANSLATE_TOP_P, TRANSLATE_MAX_RETRIES,
    ANALYZE_TEMPERATURE, ANALYZE_MAX_TOKENS, ANALYZE_TOP_P,
    DECIDE_TEMPERATURE, DECIDE_MAX_TOKENS, DECIDE_TOP_P,
)

class LLMHelper:
    """LLM助手类（支持豆包、GLM（智谱）和 DeepSeek）"""
    
    def __init__(self):
        """根据配置初始化对应的LLM客户端"""
        self.provider = LLM_PROVIDER.lower()
        
        if self.provider == "doubao":
            from llm_clients.doubao_client import DoubaoClient
            from config import DOUBAO_API_KEY, DOUBAO_MODEL, DOUBAO_BASE_URL
            if not DOUBAO_API_KEY:
                raise ValueError("请在 .env 中设置DOUBAO_API_KEY")
            self.client = DoubaoClient(api_key=DOUBAO_API_KEY, model=DOUBAO_MODEL, base_url=DOUBAO_BASE_URL)
            self.model = DOUBAO_MODEL
            print(f"🤖 使用豆包大模型: {self.model}")
        elif self.provider in ("glm", "chatglm"):
            from llm_clients.chatglm_client import ChatGLMClient
            from config import CHATGLM_API_KEY, CHATGLM_MODEL, CHATGLM_BASE_URL, CHATGLM_ENABLE_THINKING
            if not CHATGLM_API_KEY:
                raise ValueError("请在 .env 中设置CHATGLM_API_KEY")
            self.client = ChatGLMClient(api_key=CHATGLM_API_KEY, model=CHATGLM_MODEL, base_url=CHATGLM_BASE_URL)
            self.enable_thinking = CHATGLM_ENABLE_THINKING
            self.model = CHATGLM_MODEL
            print(f"🤖 使用智谱GLM模型: {self.model}")
        elif self.provider == "deepseek":
            from llm_clients.deepseek_client import DeepSeekClient
            from config import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, DEEPSEEK_ENABLE_THINKING
            if not DEEPSEEK_API_KEY:
                raise ValueError("请在 .env 中设置DEEPSEEK_API_KEY")
            self.client = DeepSeekClient(api_key=DEEPSEEK_API_KEY, model=DEEPSEEK_MODEL, base_url=DEEPSEEK_BASE_URL)
            self.enable_thinking = DEEPSEEK_ENABLE_THINKING
            self.model = DEEPSEEK_MODEL
            print(f"🤖 使用DeepSeek模型: {self.model}")
        else:
            raise ValueError(f"不支持的LLM提供商: {LLM_PROVIDER}，请在config.py中设置LLM_PROVIDER为'doubao'、'glm'或'deepseek'")

    def _call_llm(self, **request_params):
        """统一注入超时/重试等通用参数后调用模型

        Args:
            **request_params: model / messages / temperature / max_tokens / top_p 等

        Returns:
            模型的响应对象
        """
        request_params.setdefault("timeout", LLM_TIMEOUT)
        request_params.setdefault("max_retries", LLM_MAX_RETRIES)
        request_params.setdefault("retry_delay", LLM_RETRY_DELAY)
        # 对于 DeepSeek，根据配置设置 thinking 参数（默认关闭，避免思考模式消耗大量 token 导致输出为空）
        if self.provider == "deepseek" and "thinking" not in request_params:
            request_params["thinking"] = {"type": "enabled" if self.enable_thinking else "disabled"}
        return self.client.chat.completions.create(**request_params)

    def translate_title(self, title: str, abstract: str = "") -> str:
        """
        使用LLM翻译论文标题，增强的提示词和错误处理
        Args:
            title: 论文英文标题
            abstract: 论文摘要，用于提供上下文（可选）
        Returns:
            str: 中文标题
        """
        max_retries = TRANSLATE_MAX_RETRIES
        retry_delay = LLM_RETRY_DELAY  # 基础重试延迟秒数（指数退避）

        # 简洁的提示词，直接要求返回中文标题
        prompt = f"""将以下计算机视觉论文标题翻译成中文。

要求：
- 专业术语、模型名称、算法名称、缩写保持英文原样（如CLIP、ViT、NeRF、3D等）
- 只输出翻译结果，不要解释

标题：{title}"""

        for attempt in range(max_retries):
            try:
                # 构建 API 请求参数
                request_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TRANSLATE_TEMPERATURE,
                    "max_tokens": TRANSLATE_MAX_TOKENS,
                    "top_p": TRANSLATE_TOP_P,
                }
                
                # 对于 glm-4.7 模型，根据配置设置 thinking 参数
                if self.provider in ("glm", "chatglm") and "glm-4.7" in self.model:
                    if self.enable_thinking:
                        request_params["thinking"] = {"type": "enabled"}
                    else:
                        request_params["thinking"] = {"type": "disabled"}
                
                response = self._call_llm(**request_params)
                translation = response.choices[0].message.content.strip()
                # 清理可能的多余内容，只保留第一行
                if '\n' in translation:
                    translation = translation.split('\n')[0].strip()
                # 确保返回的是中文
                if translation and any('\u4e00' <= char <= '\u9fff' for char in translation):
                    return translation
                else:
                    print(f"警告：第{attempt + 1}次翻译未返回中文结果，重试中...")
                    time.sleep(min(retry_delay * (2 ** attempt), 30))
            except Exception as e:
                print(f"翻译出错 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(min(retry_delay * (2 ** attempt), 30))
                continue
        
        return f"[翻译失败] {title}"

    def analyze_paper_contribution(self, title: str, abstract: str) -> dict:
        """分析论文的核心贡献，以单句话总结的形式返回
    
        Args:
            title: 论文标题
            abstract: 论文摘要
    
        Returns:
            dict: 包含分析结果的字典，只有一个键"核心贡献"，值为单句话总结
        """
        prompt = f"""用一句话（不超过50字）总结这篇论文的核心贡献，只输出总结内容。

标题：{title}
摘要：{abstract[:500] if len(abstract) > 500 else abstract}"""
        try:
            response = self._call_llm(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=ANALYZE_TEMPERATURE,
                max_tokens=ANALYZE_MAX_TOKENS,
                top_p=ANALYZE_TOP_P,
            )
            
            # 获取单句话总结
            contribution_summary = response.choices[0].message.content.strip()
            
            # 去除可能的多余内容，只保留第一句话
            if '\n' in contribution_summary:
                contribution_summary = contribution_summary.split('\n')[0].strip()
                
            # 确保长度适当
            if len(contribution_summary) > 100:
                contribution_summary = contribution_summary[:97] + '...'
                
            return {
                "核心贡献": contribution_summary
            }
            
        except Exception as e:
            print(f"分析论文贡献时出错: {str(e)}")
            return {
                "核心贡献": "分析失败"
            }

    def decide_category(self, title: str, abstract: str, candidate_categories: List[Tuple], match_details: Dict = None) -> str:
        """使用LLM从候选类别中决定最终分类
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            candidate_categories: 候选类别列表，每个元素是(category, score, subcategory)的元组
            match_details: 关键词匹配详情字典，可选
            
        Returns:
            最终决定的类别名称
        """
        # 初始化匹配详情字典，如果没有提供
        if match_details is None:
            match_details = {}
        try:
            import categories_config
            
            # 提取摘要中的关键信息
            key_info = ""
            if len(abstract) > 100:
                # 提取摘要的开头和结尾，这通常包含最重要的信息
                intro = abstract[:200] if len(abstract) > 200 else abstract
                conclusion = abstract[-200:] if len(abstract) > 400 else ""
                key_info = f"摘要开头: {intro}\n"
                if conclusion:
                    key_info += f"摘要结尾: {conclusion}\n"
            
            # 格式化候选类别信息，增加更多上下文
            detailed_candidates = ""
            for i, (category, score, subcategory) in enumerate(candidate_categories[:5], 1):
                subcategory_info = f", 子类别: {subcategory[0] if subcategory else '无'}" if subcategory else ""
                matches = match_details.get(category, [])[:3]
                match_info = f", 关键匹配: {', '.join(matches)}" if matches else ""
                detailed_candidates += f"\n{i}. {category} (得分: {score:.2f}{subcategory_info}{match_info})"
            
            # 构建增强的提示词，提供更多上下文和更精确的指导
            prompt = f"""请作为一位资深的计算机视觉领域研究专家，对以下论文进行精确分类。请仔细分析论文的核心技术和创新点。

论文标题: {title}

{key_info}
候选类别:{detailed_candidates}

{categories_config.CATEGORY_PROMPT}

分类指南:
1. 仅从上述候选类别中选择一个最能代表论文核心创新点的类别
2. 请基于论文的技术本质做出判断，而不是仅基于关键词匹配
3. 如果论文主要是将现有技术应用到特定领域，而没有显著的技术创新，请选择“领域特定视觉应用”
4. 如果论文提出了新的算法、模型或方法，请选择最能代表这一创新的技术类别
5. 如果论文涉及多个领域，请选择最能代表其核心创新的类别

请直接返回最合适的类别名称，不要有任何解释或额外文本。只返回类别名称。"""
            
            # 调用 LLM 进行分类决策（参数来自 config.py）
            response = self._call_llm(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=DECIDE_TEMPERATURE,  # 极低温度提高稳定性
                max_tokens=DECIDE_MAX_TOKENS,
                top_p=DECIDE_TOP_P,
            )
            
            # 获取分类结果
            category = response.choices[0].message.content.strip()
            
            # 验证返回的类别是否在候选类别中
            candidate_names = [c[0] for c in candidate_categories]
            
            # 如果返回的类别在候选类别中，直接返回
            if category in candidate_names:
                return category
            
            # 如果返回的类别不在候选类别中，但在预定义类别中，也返回
            if category in categories_config.CATEGORY_DISPLAY_ORDER:
                return category
            
            # 如果都不匹配，返回得分最高的候选类别
            return candidate_categories[0][0]
            
        except Exception as e:
            print(f"LLM 分类决策出错: {str(e)}")
            # 发生错误时，返回得分最高的候选类别
            return candidate_categories[0][0] if candidate_categories else "其他 (Others)"
