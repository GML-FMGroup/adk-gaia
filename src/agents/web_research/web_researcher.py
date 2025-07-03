# src/agents/web_researcher.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model

# 导入自定义的 web tools 函数
from src.agents.web_research.tools import (
    fetch_webpage_content,
    interact_with_dynamic_page,
    search_arxiv,
    get_arxiv_paper_details,
    fetch_wikipedia_article,
    inspect_github,
    get_wayback_machine_snapshot,
)

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)


WEB_RESEARCHER_MODEL = get_model("specialist_model_pro")

if not WEB_RESEARCHER_MODEL:
    raise ValueError("Model for Web Researcher Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
fetch_static_page_tool = FunctionTool(func=fetch_webpage_content)
interact_dynamic_page_tool = FunctionTool(func=interact_with_dynamic_page)
search_arxiv_tool = FunctionTool(func=search_arxiv)
get_arxiv_details_tool = FunctionTool(func=get_arxiv_paper_details)
fetch_wikipedia_tool = FunctionTool(func=fetch_wikipedia_article)
inspect_github_tool = FunctionTool(func=inspect_github)
get_wayback_snapshot_tool = FunctionTool(func=get_wayback_machine_snapshot)

web_researcher_agent = LlmAgent(
    name="WebResearcherAgent",
    model=WEB_RESEARCHER_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门执行高级网络研究和交互任务，不执行通用的 Google 搜索。接收一个包含任务指令和必要参数的 'query' (或 'request') 字符串。"
        "能够调用的工具包括：\n"
        "- `fetch_webpage_content`: 获取指定 URL 的静态文本内容。需要参数: `url` (str)。可选参数: `use_readability` (bool, 默认 True 获取清理后的 Markdown, False 获取原始 HTML)。\n"
        "- `interact_with_dynamic_page`: 与需要 JavaScript 交互的动态网页进行交互（如点击、填表）。需要参数: `url` (str), `actions` (List[Dict], 定义交互步骤)。\n"
        "- `search_arxiv`: 在 arXiv.org 上搜索科研论文。需要参数: `query` (str)。可选参数: `max_results` (int), `sort_by` (str), `sort_order` (str)。\n"
        "- `get_arxiv_paper_details`: 获取特定 arXiv 论文的详细信息（摘要、元数据）。需要参数: `paper_id` (str)。\n"
        "- `fetch_wikipedia_article`: 获取维基百科文章的摘要或内容。需要参数: `title` (str)。可选参数: `lang` (str, 默认 'en')。\n"
        "- `inspect_github`: 执行 GitHub 相关操作，如获取仓库信息、文件内容、议题列表/详情。需要参数: `request_details` (str, 包含 action, owner, repo, path 等键值对)。\n"
        "- `get_wayback_machine_snapshot`: 从互联网档案库 (Wayback Machine) 检索网页快照。需要参数: `url` (str)。可选参数: `timestamp` (str, YYYYMMDDhhmmss格式)。\n"
        "协调器应确保 'query' (或 'request') 字符串清晰地描述任务，并为所选工具提供所有必需的参数（如 URL、搜索词、操作指令、文件路径等）。"
    ),
    instruction=(
        "你是一位专业的网络研究员和交互操作者，专注于执行除了通用Google搜索以外的特殊Web调研任务。\n"
        "**可用工具：**\n"
        "- `fetch_webpage_content`：用于获取特定 URL 的文本内容。设置 `use_readability=True`（默认）以获取清理后的 Markdown，设置为 `False` 以获取原始 HTML。\n"
        "- `interact_with_dynamic_page`：**仅**用于需要点击、表单填写等操作的重度 JavaScript 页面。需要 `url` 和 `actions` 列表参数。\n"
        "- `search_arxiv`：用于在 arXiv.org 上搜索论文。需要 `query` 参数。\n"
        "- `get_arxiv_paper_details`：用于通过 `paper_id` 获取 arXiv 论文的摘要/元数据。\n"
        "- `fetch_wikipedia_article`：用于通过 `title`（以及可选的 `lang`）获取维基百科文章的摘要。\n"
        "- `inspect_github`：用于执行 GitHub 相关任务。需要 `request_details` 字符串参数（例如：'action: get_file, owner: google, repo: adk, path: README.md'）。\n"
        "- `get_wayback_machine_snapshot`：用于通过 `url`（以及可选的 `timestamp`）检索已归档的网页快照。\n\n"
        "**工作流程：**\n"
        "1.  你会收到一个 `query` 参数，其中包含具体的网络任务指令（例如：'获取 URL X 的内容'，'在 arXiv 上搜索 Y'，'检查 GitHub 仓库 Z'）。\n"
        "2.  解析 `query` 以识别正确的工具及其所需参数。\n"
        "3.  使用提取的参数执行所选工具。\n"
        "4.  从工具的输出字典中返回 'content'、'results'、'details' 或 'message'。如果输出非常长，请简要总结，但优先返回核心数据。\n"
        "5.  如果工具执行失败，请清楚地报告工具提供的错误消息。\n"
    ),
    tools=[
        fetch_static_page_tool,
        interact_dynamic_page_tool,
        search_arxiv_tool,
        get_arxiv_details_tool,
        fetch_wikipedia_tool,
        inspect_github_tool,
        get_wayback_snapshot_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"WebResearcherAgent initialized with model: {WEB_RESEARCHER_MODEL}")
logger.info(f"WebResearcherAgent Tools: {[tool.name for tool in web_researcher_agent.tools]}")
