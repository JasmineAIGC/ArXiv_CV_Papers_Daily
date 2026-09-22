# ArXiv CV Papers Daily Update

An automated system for fetching, analyzing, and organizing the latest computer vision research papers from ArXiv with AI-powered classification capabilities.

## Features

- **Automated Paper Retrieval**: Automatically fetches the latest CV papers from ArXiv
- **AI-Powered Analysis**: Uses Doubao / ChatGLM / DeepSeek for intelligent paper categorization and analysis
- **Bilingual Support**: Provides paper titles in both English and Chinese
- **Code Link Detection**: Automatically extracts GitHub repository links
- **Organized Output**: Generates well-structured Markdown reports
- **Parallel Processing**: Utilizes multi-threading for improved efficiency
- **Smart Categorization**: Classifies papers into specific research areas

## Project Structure

```
ArXiv_CV_Papers_Daily/
├── scripts/                  # 脚本目录
│   ├── main.py               # 主入口：抓取、并行分析、生成报告
│   ├── llm_helper.py         # LLM 助手：翻译、贡献分析、分类仲裁
│   ├── llm_clients/          # LLM 客户端封装
│   │   ├── chatglm_client.py # ChatGLM 客户端
│   │   ├── doubao_client.py  # 豆包客户端
│   │   └── deepseek_client.py # DeepSeek 客户端
│   ├── classifier.py         # 关键词分类与子类别判定
│   ├── text_utils.py         # NLTK 初始化与文本预处理
│   ├── markdown_output.py    # Markdown 生成与保存
│   ├── categories_config.py  # 分类体系与关键词配置
│   ├── config.py             # 非敏感配置（可提交 Git）
│   ├── .env.example          # 密钥模板（复制为 .env）
│   └── requirements.txt      # 依赖
├── data/                     # 表格格式论文信息
│   └── YYYY-MM/
│       └── YYYY-MM-DD.md
└── local/                    # 详细格式论文信息（本地，不提交 Git）
    └── YYYY-MM/
        └── YYYY-MM-DD.md
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Doubao / ChatGLM / DeepSeek API key（配置在 `scripts/.env`）
- Stable internet connection

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure API keys via `scripts/.env`（见下方 Configuration）

## Configuration

非敏感配置集中在 `config.py`（可安全提交到 Git），所有参数均会生效：

- `LLM_PROVIDER` / `DOUBAO_MODEL` / `DOUBAO_BASE_URL` / `CHATGLM_MODEL` / `CHATGLM_BASE_URL` / `CHATGLM_ENABLE_THINKING` / `DEEPSEEK_MODEL` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_ENABLE_THINKING`：LLM 提供商与模型
- `LLM_TIMEOUT` / `LLM_MAX_RETRIES` / `LLM_RETRY_DELAY` / `LLM_MAX_WORKERS`：LLM 请求通用参数
- `TRANSLATE_*` / `ANALYZE_*` / `DECIDE_*`：各任务生成参数（temperature / max_tokens / top_p / 重试次数）
- `ENABLE_TITLE_TRANSLATION` / `ENABLE_CONTRIBUTION_ANALYSIS` / `ENABLE_LLM_ARBITRATION` / `ENABLE_DETAILED_OUTPUT`：功能开关
- `QUERY_DAYS_AGO`: Days to look back (0=today, 1=yesterday)
- `MAX_RESULTS` / `MAX_WORKERS` / `MAX_AUTHORS_SHOWN`：论文数量、并发线程数、作者展示上限
- `ARXIV_QUERY` / `ARXIV_PAGE_SIZE` / `ARXIV_DELAY_SECONDS` / `ARXIV_NUM_RETRIES` / `ARXIV_BATCH_SIZE`：ArXiv 客户端参数
- `DATA_DIR` / `LOCAL_DIR`：输出目录（相对于 scripts/）

敏感配置（API Key）放在 `scripts/.env` 中（已被 .gitignore 忽略，不会提交）：

1. 复制模板：`cp scripts/.env.example scripts/.env`
2. 填入真实密钥：
```bash
DOUBAO_API_KEY=your_doubao_api_key_here
CHATGLM_API_KEY=your_chatglm_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```
同名环境变量优先级高于 `.env` 文件。

## Usage

Run the main script:
```bash
python main.py
```

### Output Files

The script generates two types of Markdown files:

1. Table Format (`data/YYYY-MM/YYYY-MM-DD.md`):
   - Basic paper information
   - Categorized by research areas
   - Concise tabular format

2. Detailed Format (`local/YYYY-MM/YYYY-MM-DD.md`):
   - Comprehensive paper details
   - AI-generated analysis
   - Core contributions
   - Code links

## Research Categories

分类体系定义在 `categories_config.py`，共 13 个一级类别（中英文对照）：

1. 视觉表征与基础模型 (Visual Representation & Foundation Models)
2. 视觉识别与理解 (Visual Recognition & Understanding)
3. 生成式视觉模型 (Generative Visual Modeling)
4. 三维视觉与几何推理 (3D Vision & Geometric Reasoning)
5. 时序视觉分析 (Temporal Visual Analysis)
6. 自监督与表征学习 (Self-supervised & Representation Learning)
7. 计算效率与模型优化 (Computational Efficiency & Model Optimization)
8. 鲁棒性与可靠性 (Robustness & Reliability)
9. 低资源与高效学习 (Low-resource & Efficient Learning)
10. 具身智能与交互视觉 (Embodied Intelligence & Interactive Vision)
11. 视觉-语言协同理解 (Vision-Language Joint Understanding)
12. 领域特定视觉应用 (Domain-specific Visual Applications)
13. 新兴理论与跨学科方向 (Emerging Theory & Interdisciplinary Directions)
14. 其他 (Others)

## Automated Deployment

Set up daily automatic runs using crontab:
```bash
# Edit crontab
crontab -e

# Add this line to run at 9 AM daily
0 9 * * * cd /path/to/ArXiv_CV_Papers_Daily/scripts && python main.py
```

## Error Handling

The script includes:
- ArXiv API rate limit handling
- Network error recovery
- Parallel processing error handling
- File system error handling

## Security Notes

- API Key 存放在 `scripts/.env`（已被 .gitignore 忽略），或通过同名环境变量注入
- 定期轮换 API Key，监控 API 使用量
- Use virtual environments

## Future Improvements

- 更好代码链接检测
- Interactive web interface
- Advanced paper filtering
- 交互式网页界面
- 多语言支持
- 高级论文筛选
