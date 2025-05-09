# src/agents/web_researcher.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model

# 导入自定义的 web tools 函数
from src.tools.web_tools import (
    fetch_webpage_content,
    interact_with_dynamic_page,
    search_arxiv,
    get_arxiv_paper_details,
    fetch_wikipedia_article,
    inspect_github,
    get_wayback_machine_snapshot,
)

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
    description=(
        "Specializes in ADVANCED web tasks: fetching static/dynamic webpages, interacting with pages (clicks, forms), "
        "searching arXiv, Wikipedia, GitHub, and Wayback Machine. Does NOT perform general Google searches."
    ),
    instruction=(
        "You are an expert web researcher and interactor, focusing on specific tasks beyond general search.\n"
        "**Available Tools (FunctionTool only):**\n"
        "- `fetch_webpage_content`: Use to get the text content of a specific URL. Set `use_readability=True` (default) for cleaned Markdown, `False` for raw HTML.\n"
        "- `interact_with_dynamic_page`: Use ONLY for JavaScript-heavy pages needing clicks, form filling, etc. Requires `url` and `actions` list.\n"
        "- `search_arxiv`: Use for searching papers on arXiv.org. Requires `query`.\n"
        "- `get_arxiv_paper_details`: Use to get abstract/metadata for an arXiv paper via `paper_id`.\n"
        "- `fetch_wikipedia_article`: Use to get the summary of a Wikipedia article via `title` (and optional `lang`).\n"
        "- `inspect_github`: Use for GitHub tasks. Requires `request_details` string (e.g., 'action: get_file, owner: google, repo: adk, path: README.md').\n"
        "- `get_wayback_machine_snapshot`: Use to retrieve archived webpages via `url` (and optional `timestamp`).\n\n"
        "**Workflow:**\n"
        "1.  You will receive a `query` argument containing the specific web task instructions (e.g., 'Fetch content from URL X', 'Search arXiv for Y', 'Inspect GitHub repo Z').\n"
        "2.  Parse the `query` to identify the correct tool and its required parameters.\n"
        "3.  Execute the chosen tool with the extracted parameters.\n"
        "4.  Return the 'content', 'results', 'details', or 'message' from the tool's output dictionary. Summarize briefly if the output is very long, but prioritize returning the core data.\n"
        "5.  If a tool fails, clearly report the error message provided by the tool.\n"
        "6.  **DO NOT** attempt general web searches; that is handled by a different agent."
    ),
    tools=[
        fetch_static_page_tool,
        interact_dynamic_page_tool,
        search_arxiv_tool,
        get_arxiv_details_tool,
        fetch_wikipedia_tool,
        inspect_github_tool,
        get_wayback_snapshot_tool,
    ]
)

logger.info(f"WebResearcherAgent initialized with model: {WEB_RESEARCHER_MODEL}")
logger.info(f"WebResearcherAgent Tools: {[tool.name for tool in web_researcher_agent.tools]}")