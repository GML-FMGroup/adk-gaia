# src/agents/google_search_agent.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from src.core.config import get_model

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_MODEL = get_model("specialist_model_flash")

if not GOOGLE_SEARCH_MODEL:
    raise ValueError("Model for GoogleSearchAgent not found in configuration.")

google_search_agent = LlmAgent(
    name="GoogleSearchAgent",
    model=GOOGLE_SEARCH_MODEL,
    description=(
        "Specializes EXCLUSIVELY in performing web searches using the built-in Google Search tool. "
        "Use this for general knowledge lookups, current events, or when a direct web search is needed."
    ),
    instruction=(
        "You are a dedicated web search agent. Your ONLY function is to use the `google_search` tool. "
        "You will receive a single string argument named `query` containing the search term(s). "
        "Execute the search using the `google_search` tool with the provided query. "
        "Return ONLY the search results provided by the tool. Do not add any explanation or summary. "
        "Do not attempt any other action."
    ),
    tools=[google_search] # Only the built-in tool
)

logger.info(f"GoogleSearchAgent initialized with model: {GOOGLE_SEARCH_MODEL}")
logger.info(f"GoogleSearchAgent Tools: {[tool.name for tool in google_search_agent.tools]}")