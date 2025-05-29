# src/agents/google_search_agent.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from src.core.config import get_model

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_MODEL = get_model("specialist_model_flash")

if not GOOGLE_SEARCH_MODEL:
    raise ValueError("Model for GoogleSearchAgent not found in configuration.")

google_search_agent = LlmAgent(
    name="GoogleSearchAgent",
    model=GOOGLE_SEARCH_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门执行 Google 网页搜索。接收一个包含搜索关键词的 'request' 字符串 (该字符串将作为搜索查询)。"
        "使用 ADK 内置的 `google_search` 工具来获取搜索结果。"
        "适用于需要获取一般知识、查找当前事件信息或任何需要直接进行网络搜索的场景。"
        "此智能体只负责执行搜索并返回原始搜索结果（通常是摘要或片段列表），不进行深度分析或内容提取。"
        "如果需要从特定网页提取详细内容，应使用 `WebResearcherAgent`。"
    ),
    instruction=(
        "你是一个专门的Google搜索智能体。你唯一的功能就是使用 `google_search` 工具。"
        "你会收到一个名为 `query` 的字符串参数，其中包含搜索词。"
        "使用提供的查询通过 `google_search` 工具执行搜索。"
        "返回工具提供的搜索结果和原始网页的URL。"
        "当呈现来自搜索结果的事实时，如果查询暗示了顺序，请尽量忠于来源并按实体出现的顺序列出它们。" # MODIFIED
        "如果同一查询存在多个不同的事实或数字，请清晰地呈现它们，并在可能的情况下指出任何歧义。"
    ),
    tools=[google_search], # Only the built-in tool
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"GoogleSearchAgent initialized with model: {GOOGLE_SEARCH_MODEL}")
logger.info(f"GoogleSearchAgent Tools: {[tool.name for tool in google_search_agent.tools]}")