"""
项目配置文件（不含任何密钥，可提交到 Git）

敏感信息（API Key 等）请放在同目录下的 .env 文件中（该文件已被 .gitignore 忽略），
格式参考 .env.example。也可以直接设置同名环境变量，环境变量优先级高于 .env。
"""
import os


def _load_dotenv():
    """极简 .env 加载器：读取 KEY=VALUE 行并注入环境变量（不覆盖已存在的变量）"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()

# =============================================================================
# LLM 提供商配置 —— 开关参数：控制使用哪个大模型
# 在 .env 中设置 LLM_PROVIDER（同名环境变量优先级高于 .env）
# 可选值: "doubao"（豆包）、"glm"（智谱 GLM，旧名 "chatglm" 同样可用）或 "deepseek"（DeepSeek）
# =============================================================================
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "doubao")  # 在 .env 中改为 "glm" 或 "deepseek" 切换模型

# -----------------------------------------------------------------------------
# 豆包（ByteDance Doubao）配置
# 仅在 LLM_PROVIDER = "doubao" 时生效
# -----------------------------------------------------------------------------
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "")  # 豆包 API Key（在 .env 中配置）
DOUBAO_MODEL = os.environ.get("DOUBAO_MODEL", "doubao-1-5-lite-32k-250115")   # 豆包模型名称（在 .env 中配置）
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com"   # 豆包 API 地址

# -----------------------------------------------------------------------------
# GLM（智谱AI）配置
# 仅在 LLM_PROVIDER = "glm"（或旧名 "chatglm"）时生效
# -----------------------------------------------------------------------------
CHATGLM_API_KEY = os.environ.get("CHATGLM_API_KEY", "")      # GLM API Key（在 .env 中配置）
CHATGLM_MODEL = os.environ.get("CHATGLM_MODEL", "glm-5.1")   # GLM 模型名称（在 .env 中配置）
CHATGLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"   # GLM 接口地址
CHATGLM_ENABLE_THINKING = False                               # 是否启用 glm-4.7 thinking 模式

# -----------------------------------------------------------------------------
# DeepSeek（深度求索）配置
# 仅在 LLM_PROVIDER = "deepseek" 时生效
# -----------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")                          # DeepSeek API Key（在 .env 中配置）
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")                # DeepSeek 模型名称（deepseek-flash 即 DeepSeek-V4.1-Flash）
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # DeepSeek API 地址
DEEPSEEK_ENABLE_THINKING = os.environ.get("DEEPSEEK_ENABLE_THINKING", "False").lower() in ("1", "true", "yes")  # 是否启用 thinking（默认关闭，翻译/分类任务建议关闭）

# =============================================================================
# LLM 请求通用参数（对所有任务生效）
# =============================================================================
LLM_TIMEOUT = 60           # 单次请求超时（秒）
LLM_MAX_RETRIES = 3        # 单次请求失败最大重试次数
LLM_RETRY_DELAY = 2        # 重试基础延迟（秒），按指数退避增长
LLM_MAX_WORKERS = 8        # LLM 任务并发线程数（翻译/分析等共享线程池）

# =============================================================================
# 各任务 LLM 生成参数（temperature / max_tokens / top_p）
# =============================================================================
# 标题翻译
TRANSLATE_TEMPERATURE = 0.1
TRANSLATE_MAX_TOKENS = 150
TRANSLATE_TOP_P = 0.7
TRANSLATE_MAX_RETRIES = 10    # 翻译结果不含中文时的重试次数

# 核心贡献分析
ANALYZE_TEMPERATURE = 0.3
ANALYZE_MAX_TOKENS = 200
ANALYZE_TOP_P = 0.7

# 分类仲裁（候选类别得分接近时的LLM决策）
DECIDE_TEMPERATURE = 0.01
DECIDE_MAX_TOKENS = 50
DECIDE_TOP_P = 0.3

# =============================================================================
# 功能开关
# =============================================================================
ENABLE_TITLE_TRANSLATION = True       # 是否调用LLM翻译标题（关闭时使用英文标题）
ENABLE_CONTRIBUTION_ANALYSIS = True   # 是否调用LLM分析核心贡献
ENABLE_LLM_ARBITRATION = True         # 分类歧义时是否用LLM仲裁（关闭时取得分最高类别）
ENABLE_DETAILED_OUTPUT = True         # 是否生成 local/ 详细版 Markdown

# =============================================================================
# ArXiv 抓取参数
# =============================================================================
QUERY_DAYS_AGO = 1    # 查询几天前的论文：0=今天，1=昨天，2=前天
MAX_RESULTS = 300     # 单次最多抓取的论文数量
MAX_WORKERS = 4       # 论文处理并发线程数

# ArXiv 客户端参数（一般无需修改）
ARXIV_QUERY = "cat:cs.CV"    # ArXiv 搜索类别
ARXIV_PAGE_SIZE = 100        # 每页返回数量
ARXIV_DELAY_SECONDS = 3.0    # 请求间隔（秒），arXiv 官方要求 >=3 秒，过快会被限流返回 HTTP 406
ARXIV_NUM_RETRIES = 5        # 失败重试次数
ARXIV_BATCH_SIZE = 10        # 每批处理论文数
MAX_AUTHORS_SHOWN = 8        # 每篇论文展示的作者数量上限

# =============================================================================
# 输出路径
# 相对于 scripts/ 目录的路径
# =============================================================================
DATA_DIR = "../data"    # 表格格式 Markdown 输出目录
LOCAL_DIR = "../local"  # 详细格式 Markdown 输出目录（本地，不提交 Git）
