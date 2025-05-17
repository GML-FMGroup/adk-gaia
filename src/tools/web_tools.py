# src/tools/web_tools.py
import os
import json
import logging
import requests
from bs4 import BeautifulSoup
import markdownify
import datetime
from typing import Dict, Any, Optional, List

# Playwright (异步，用于动态页面)
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
    logging.info("Playwright found.")
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not installed (`pip install playwright && playwright install`). Dynamic web scraping disabled.")

# arXiv
try:
    import arxiv
    ARXIV_AVAILABLE = True
    logging.info("arXiv library found.")
except ImportError:
    ARXIV_AVAILABLE = False
    logging.warning("arXiv library not installed (`pip install arxiv`). arXiv search disabled.")

# Wikipedia
try:
    import wikipediaapi
     # 使用 wikipedia-api 库
    WIKIPEDIA_AVAILABLE = True
    logging.info("wikipedia-api library found.")
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    logging.warning("wikipedia-api library not installed (`pip install wikipedia-api`). Wikipedia search disabled.")

# GitHub (使用 PyGithub)
try:
    from github import Github, GithubException, UnknownObjectException
    PYGITHUB_AVAILABLE = True
    logging.info("PyGithub library found.")
    # 从环境变量获取 GitHub Token
    GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not GITHUB_TOKEN:
        logging.warning("GITHUB_PERSONAL_ACCESS_TOKEN environment variable not set. GitHub tool may fail for private repos or rate limits.")
        gh_client = Github() # 未认证的客户端，速率有限
    else:
        gh_client = Github(GITHUB_TOKEN)
except ImportError:
    PYGITHUB_AVAILABLE = False
    logging.warning("PyGithub library not installed (`pip install PyGithub`). GitHub inspection disabled.")
    gh_client = None

# waybackpy 相关的导入
try:
    import waybackpy
    from waybackpy import WaybackMachineCDXServerAPI # 重要：使用CDXServerAPI
    from waybackpy.exceptions import NoCDXRecordFound, WaybackError
    WAYBACKPY_AVAILABLE = True
    WAYBACKPY_VERSION = waybackpy.__version__ # 可选，用于调试
    logger = logging.getLogger(__name__) # 假设logger已在模块级别定义
    USER_AGENT = "ADKGaiaSolver/1.0 (YourProject/YourContact)"
    logger.info(f"Waybackpy version {WAYBACKPY_VERSION} loaded.")
except ImportError:
    WAYBACKPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("waybackpy library not found. Wayback Machine functionality will be disabled.")
    # 定义虚拟类以便代码在缺少库时仍能解析
    class WaybackMachineCDXServerAPI: pass
    class NoCDXRecordFound(Exception): pass
    class WaybackError(Exception): pass

# Readabilipy (用于静态页面清理)
try:
    import readabilipy
    READABILIPY_AVAILABLE = True
    logging.info("Readabilipy library found.")
except ImportError:
    READABILIPY_AVAILABLE = False
    logging.warning("Readabilipy library not installed (`pip install readabilipy`). Static web scraping quality may be reduced.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _clean_html_to_markdown(html: str) -> str:
    """Cleans HTML and converts to Markdown using Readabilipy and Markdownify."""
    if not READABILIPY_AVAILABLE:
        # Fallback: Basic cleaning with BeautifulSoup if readabilipy is missing
        soup = BeautifulSoup(html, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        # Get text, preserving some structure with newlines might be better
        text = soup.get_text(separator = "\n", strip=True)
        return text # Return plain text as fallback

    try:
        article = readabilipy.simple_json_from_html_string(html, use_readability=True)
        if not article or not article.get('content'):
            logger.warning("Readabilipy could not extract main content. Falling back to raw text.")
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator='\n', strip=True)

        # Convert cleaned HTML to Markdown
        markdown_content = markdownify.markdownify(article['content'], heading_style="ATX")
        return markdown_content
    except Exception as e:
        logger.error(f"Error during HTML cleaning/conversion: {e}", exc_info=True)
        # Fallback to simple text extraction on error
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n', strip=True)

# --- Tool Implementations ---

def fetch_webpage_content(url: str, use_readability: bool = True) -> Dict[str, Any]:
    """
    Fetches the content of a given URL and returns it, optionally cleaned and converted to Markdown.

    Args:
        url (str): The URL of the webpage to fetch.
        use_readability (bool): If True (default), attempts to extract the main article content
                                 and convert it to Markdown. If False, returns the raw HTML.

    Returns:
        dict: A dictionary with 'status' and 'content' or 'message'.
              Content will be Markdown (if use_readability=True and successful) or raw HTML/text.
    """
    logger.info(f"Attempting to fetch static content from URL: {url} (Readability: {use_readability})")
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        content_type = response.headers.get('content-type', '').lower()

        if 'html' in content_type:
            raw_html = response.text
            if use_readability:
                content = _clean_html_to_markdown(raw_html)
                logger.info(f"Successfully fetched and cleaned content from {url}")
            else:
                content = raw_html # Return raw HTML if readability is off
                logger.info(f"Successfully fetched raw HTML from {url}")
        elif 'text' in content_type:
            content = response.text # Return plain text directly
            logger.info(f"Successfully fetched plain text from {url}")
        else:
             logger.warning(f"Fetched content from {url} with non-HTML/text type: {content_type}. Returning raw content.")
             content = response.text # Or response.content for binary? Decide based on need.

        # Truncate long content to prevent exceeding context limits
        max_len = 100000 # Increased limit for web content
        if len(content) > max_len:
             logger.warning(f"Fetched content from {url} truncated to {max_len} characters.")
             content = content[:max_len] + f"\n... (truncated, original size ~{len(content)} chars)"

        return {"status": "success", "content": content}

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return {"status": "error", "message": f"Failed to fetch URL {url}: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


async def interact_with_dynamic_page(url: str, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Interacts with a dynamic webpage using Playwright. Executes a sequence of actions.

    Args:
        url (str): The initial URL to navigate to.
        actions (List[Dict[str, Any]]): A list of action dictionaries. Each dictionary must have an 'action_type' key
                                         and necessary arguments for that action. Supported action_types:
                                         - 'wait': {'action_type': 'wait', 'milliseconds': 500}
                                         - 'click': {'action_type': 'click', 'selector': 'css_selector'}
                                         - 'fill': {'action_type': 'fill', 'selector': 'css_selector', 'value': 'text_to_fill'}
                                         - 'get_text': {'action_type': 'get_text', 'selector': 'css_selector_optional'} (gets body text if no selector)
                                         - 'get_html': {'action_type': 'get_html', 'selector': 'css_selector'}
                                         - 'screenshot': {'action_type': 'screenshot', 'filename': 'screenshot.png'} (returns success/failure)

    Returns:
        dict: Dictionary with 'status' ('success' or 'error') and 'result' (text/html content or status message) or 'message'.
              The 'result' key contains the output of the *last* action in the list if it produces output (like get_text/get_html).
    """
    logger.info(f"Attempting dynamic interaction with URL: {url} actions: {actions}")
    if not PLAYWRIGHT_AVAILABLE:
        return {"status": "error", "message": "Playwright is not installed. Cannot interact with dynamic pages."}

    async with async_playwright() as p:
        browser = None
        try:
            # Launch browser (consider headless=True for servers)
            browser = await p.chromium.launch(headless=True) # Changed to headless=True
            page = await browser.new_page()
            logger.info(f"Navigating to {url}")
            await page.goto(url, wait_until='networkidle', timeout=30000) # Increased timeout and wait condition

            last_result = f"Successfully navigated to {url}." # Default result

            for i, action_data in enumerate(actions):
                action_type = action_data.get("action_type")
                logger.info(f"Executing action {i+1}/{len(actions)}: {action_type} with args {action_data}")

                if action_type == "wait":
                    milliseconds = action_data.get("milliseconds", 500)
                    await page.wait_for_timeout(milliseconds)
                    last_result = f"Waited for {milliseconds} ms."
                elif action_type == "click":
                    selector = action_data.get("selector")
                    if not selector: return {"status": "error", "message": f"Action {i+1} 'click' requires a 'selector'."}
                    await page.locator(selector).click(timeout=10000)
                    # Wait for potential navigation or content changes
                    await page.wait_for_load_state('networkidle', timeout=15000)
                    last_result = f"Clicked element '{selector}'."
                elif action_type == "fill":
                    selector = action_data.get("selector")
                    value = action_data.get("value")
                    if not selector or value is None: return {"status": "error", "message": f"Action {i+1} 'fill' requires 'selector' and 'value'."}
                    await page.locator(selector).fill(value, timeout=10000)
                    last_result = f"Filled element '{selector}'."
                elif action_type == "get_text":
                    selector = action_data.get("selector")
                    if selector:
                        # Get text from specific element, handle potential multiple matches
                        elements = page.locator(selector)
                        count = await elements.count()
                        if count > 0:
                            texts = await elements.all_text_contents()
                            last_result = "\n".join(texts)
                        else:
                            last_result = f"No element found with selector '{selector}'."
                    else:
                        # Get text from body
                        last_result = await page.locator('body').inner_text(timeout=10000)
                    logger.info(f"Get text result length: {len(last_result)}")
                elif action_type == "get_html":
                    selector = action_data.get("selector")
                    if not selector: return {"status": "error", "message": f"Action {i+1} 'get_html' requires a 'selector'."}
                    # Get HTML from specific element
                    elements = page.locator(selector)
                    count = await elements.count()
                    if count > 0:
                       last_result = await elements.first.inner_html(timeout=10000)
                    else:
                       last_result = f"No element found with selector '{selector}'."
                    logger.info(f"Get HTML result length: {len(last_result)}")
                elif action_type == "screenshot":
                     filename = action_data.get("filename", "screenshot.png")
                     # Ensure filename is safe? Maybe restrict path? For now, just filename.
                     safe_filename = os.path.basename(filename)
                     await page.screenshot(path=safe_filename, full_page=True)
                     last_result = f"Screenshot saved as {safe_filename}." # Note: Agent can't access this file directly
                else:
                    return {"status": "error", "message": f"Unsupported action_type: {action_type}"}

            # Truncate the final result if it's text/html
            if isinstance(last_result, str):
                max_len = 15000
                if len(last_result) > max_len:
                     logger.warning(f"Dynamic interaction result truncated to {max_len} characters.")
                     last_result = last_result[:max_len] + f"\n... (truncated, original size ~{len(last_result)} chars)"

            await browser.close()
            logger.info(f"Successfully completed dynamic interaction for {url}")
            return {"status": "success", "result": last_result}

        except Exception as e:
            logger.error(f"Error during Playwright interaction with {url}: {e}", exc_info=True)
            if browser:
                await browser.close()
            return {"status": "error", "message": f"Playwright interaction failed: {str(e)}"}


def search_arxiv(query: str, max_results: int = 5, sort_by: str = 'submittedDate', sort_order: str = 'descending') -> Dict[str, Any]:
    """
    Searches arXiv for papers matching the query.

    Args:
        query (str): The search query (keywords, author, title, etc.).
        max_results (int): Maximum number of results to return (default 5, max 50).
        sort_by (str): Field to sort by ('relevance', 'lastUpdatedDate', 'submittedDate'). Default 'submittedDate'.
        sort_order (str): Sort order ('ascending', 'descending'). Default 'descending'.

    Returns:
        dict: Dictionary with 'status' and 'results' (list of paper summaries) or 'message'.
              Each paper summary includes title, authors, published date, abstract snippet, id, pdf_url.
    """
    logger.info(f"Searching arXiv with query: '{query}', max_results={max_results}")
    if not ARXIV_AVAILABLE:
        return {"status": "error", "message": "arXiv library not available."}

    try:
        # Validate sort_by and sort_order
        sort_by_map = {
            'relevance': arxiv.SortCriterion.Relevance,
            'lastupdateddate': arxiv.SortCriterion.LastUpdatedDate,
            'submitteddate': arxiv.SortCriterion.SubmittedDate
        }
        sort_order_map = {
            'ascending': arxiv.SortOrder.Ascending,
            'descending': arxiv.SortOrder.Descending
        }
        sort_criterion = sort_by_map.get(sort_by.lower(), arxiv.SortCriterion.SubmittedDate)
        sort_order_enum = sort_order_map.get(sort_order.lower(), arxiv.SortOrder.Descending)

        safe_max_results = min(max(1, max_results), 50) # Limit results

        search = arxiv.Search(
            query=query,
            max_results=safe_max_results,
            sort_by=sort_criterion,
            sort_order=sort_order_enum
        )

        client = arxiv.Client()
        results_list = []
        for result in client.results(search):
             summary = {
                 "id": result.get_short_id(),
                 "title": result.title,
                 "authors": [author.name for author in result.authors],
                 "published": result.published.strftime('%Y-%m-%d'),
                 "abstract": result.summary[:300] + "..." if result.summary else "N/A", # Snippet
                 "pdf_url": result.pdf_url,
                 "primary_category": result.primary_category,
                 "categories": result.categories
             }
             results_list.append(summary)

        logger.info(f"arXiv search successful, found {len(results_list)} results.")
        return {"status": "success", "results": results_list}

    except Exception as e:
        logger.error(f"Error searching arXiv: {e}", exc_info=True)
        return {"status": "error", "message": f"arXiv search failed: {str(e)}"}

# Note: get_arxiv_paper_details might be complex. A simple version:
def get_arxiv_paper_details(paper_id: str) -> Dict[str, Any]:
    """
    Fetches the abstract and metadata for a specific arXiv paper by its ID.
    Full text fetching/parsing is complex and not implemented here.

    Args:
        paper_id (str): The arXiv ID (e.g., '2310.06825', 'math/0309136v1').

    Returns:
        dict: Dictionary with 'status' and 'details' or 'message'.
              Details include title, authors, abstract, categories, published date, pdf_url.
    """
    logger.info(f"Fetching details for arXiv paper ID: {paper_id}")
    if not ARXIV_AVAILABLE:
        return {"status": "error", "message": "arXiv library not available."}
    try:
        search = arxiv.Search(id_list=[paper_id])
        client = arxiv.Client()
        paper = next(client.results(search)) # Get the first (and only) result

        if not paper:
             return {"status": "error", "message": f"Paper with ID '{paper_id}' not found."}

        details = {
            "id": paper.get_short_id(),
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "abstract": paper.summary,
            "categories": paper.categories,
            "published": paper.published.isoformat(), # Use ISO format
            "updated": paper.updated.isoformat(),
            "pdf_url": paper.pdf_url,
            "comment": paper.comment
        }
        logger.info(f"Successfully fetched details for arXiv paper: {paper_id}")
        return {"status": "success", "details": details}

    except StopIteration:
         logger.error(f"Paper with ID '{paper_id}' not found on arXiv.")
         return {"status": "error", "message": f"Paper with ID '{paper_id}' not found."}
    except Exception as e:
        logger.error(f"Error fetching arXiv paper details for {paper_id}: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to fetch arXiv paper details: {str(e)}"}


def fetch_wikipedia_article(title: str, lang: str = "en") -> Dict[str, Any]:
    """
    Fetches the summary or full text content of a Wikipedia article.

    Args:
        title (str): The title of the Wikipedia article.
        lang (str): The language code for Wikipedia (e.g., 'en', 'es', 'de'). Default 'en'.

    Returns:
        dict: Dictionary with 'status' and 'content' (summary or partial text) or 'message'.
    """
    logger.info(f"Fetching Wikipedia article: '{title}' (lang={lang})")
    if not WIKIPEDIA_AVAILABLE:
        return {"status": "error", "message": "wikipedia-api library not available."}
    try:
        wiki_wiki = wikipediaapi.Wikipedia(USER_AGENT, lang)
        page = wiki_wiki.page(title)

        if not page.exists():
            logger.warning(f"Wikipedia page '{title}' (lang={lang}) not found.")
            # Attempt search suggestion
            # Note: wikipedia-api doesn't have a direct search suggestion.
            # A fallback could be to use the search method, but it returns titles.
            return {"status": "error", "message": f"Wikipedia page '{title}' in language '{lang}' not found."}

        # Return summary first, potentially adding full text if needed and short enough?
        # For GAIA, often the summary or specific sections are needed. Let's return summary.
        # Full text can be very long.
        content = page.summary
        if not content: # Sometimes summary might be empty
             content = page.text[:5000] + "... (full text truncated)" # Get beginning of full text as fallback

        logger.info(f"Successfully fetched summary for Wikipedia article: '{title}' (lang={lang})")

        # Truncate if necessary
        max_len = 15000
        if len(content) > max_len:
            logger.warning(f"Wikipedia content for '{title}' truncated.")
            content = content[:max_len] + f"\n... (truncated, original summary/text was longer)"


        return {"status": "success", "title": page.title, "summary": content, "url": page.fullurl}
    except Exception as e:
        logger.error(f"Error fetching Wikipedia article '{title}': {e}", exc_info=True)
        return {"status": "error", "message": f"Wikipedia fetch failed: {str(e)}"}

# Wikidata fetch might be too specialized, skipping for now unless GAIA tasks specifically require it.

def inspect_github(request_details: str) -> Dict[str, Any]:
    """
    Inspects GitHub based on provided details. Can fetch repo info, file content, issue list/details, or search code.
    The request_details string must specify the 'action' and necessary parameters like 'owner', 'repo', 'path', 'issue_number', 'query'.
    Example request_details: "action: get_repo, owner: google, repo: adk"
    Example request_details: "action: get_file, owner: google, repo: adk, path: README.md"
    Example request_details: "action: list_issues, owner: google, repo: adk, state: open"
    Example request_details: "action: get_issue, owner: google, repo: adk, issue_number: 1"
    Example request_details: "action: search_code, query: 'google-adk language:python', repo: google/adk"

    Args:
        request_details (str): A string containing key-value pairs describing the desired action and parameters.
                               Keys: action, owner, repo, path, issue_number, state, query, ref (for branch/tag/sha).

    Returns:
        dict: Dictionary with 'status' and 'content' or 'message'. Content varies based on action.
    """
    logger.info(f"Attempting GitHub inspection with details: {request_details}")
    if not PYGITHUB_AVAILABLE or not gh_client:
        return {"status": "error", "message": "GitHub library (PyGithub) not available or not authenticated."}

    # Simple parsing of the request string (can be improved with regex or more robust parsing)
    params = {}
    try:
        parts = request_details.split(',')
        for part in parts:
            key_value = part.split(':', 1)
            if len(key_value) == 2:
                key = key_value[0].strip().lower().replace('-', '_') # Normalize key
                value = key_value[1].strip()
                # Basic type inference (improve if needed)
                if key == 'issue_number':
                    params[key] = int(value)
                else:
                    params[key] = value
            else:
                 logger.warning(f"Could not parse part: '{part}' in GitHub request.")
    except Exception as parse_err:
         logger.error(f"Error parsing request_details string '{request_details}': {parse_err}")
         return {"status": "error", "message": "Could not parse request_details string."}


    action = params.get("action")
    owner = params.get("owner")
    repo = params.get("repo")

    if not action:
        return {"status": "error", "message": "Missing 'action' in request_details."}

    try:
        # --- Get Repository Info ---
        if action == "get_repo":
            if not owner or not repo: return {"status": "error", "message": "Action 'get_repo' requires 'owner' and 'repo'."}
            repo_obj = gh_client.get_repo(f"{owner}/{repo}")
            content = {
                "full_name": repo_obj.full_name,
                "description": repo_obj.description,
                "stars": repo_obj.stargazers_count,
                "forks": repo_obj.forks_count,
                "language": repo_obj.language,
                "url": repo_obj.html_url,
                "default_branch": repo_obj.default_branch,
            }
            logger.info(f"Fetched repo info for {owner}/{repo}")
            return {"status": "success", "content": json.dumps(content, indent=2)}

        # --- Get File Content ---
        elif action == "get_file":
            path = params.get("path")
            ref = params.get("ref") # Optional branch/tag/sha
            if not owner or not repo or not path: return {"status": "error", "message": "Action 'get_file' requires 'owner', 'repo', and 'path'."}
            repo_obj = gh_client.get_repo(f"{owner}/{repo}")
            try:
                file_content = repo_obj.get_contents(path, ref=ref)
                if isinstance(file_content, list): # It's a directory
                     dir_list = [{"name": item.name, "path": item.path, "type": item.type, "sha": item.sha} for item in file_content]
                     logger.info(f"Fetched directory listing for {owner}/{repo}/{path}")
                     return {"status": "success", "content_type": "directory", "content": json.dumps(dir_list, indent=2)}
                else: # It's a file
                     decoded_content = file_content.decoded_content.decode('utf-8')
                     logger.info(f"Fetched file content for {owner}/{repo}/{path}")
                     # Truncate long files
                     max_len = 15000
                     if len(decoded_content) > max_len:
                         logger.warning(f"GitHub file content truncated.")
                         decoded_content = decoded_content[:max_len] + "\n... (truncated)"
                     return {"status": "success", "content_type": "file", "content": decoded_content, "sha": file_content.sha}
            except UnknownObjectException:
                 logger.error(f"File or directory not found: {owner}/{repo}/{path} (ref={ref})")
                 return {"status": "error", "message": f"File or directory not found at path '{path}'."}

        # --- List Issues ---
        elif action == "list_issues":
            if not owner or not repo: return {"status": "error", "message": "Action 'list_issues' requires 'owner' and 'repo'."}
            state = params.get("state", "open") # Default to open issues
            repo_obj = gh_client.get_repo(f"{owner}/{repo}")
            issues = repo_obj.get_issues(state=state) # Paginated result
            issue_list = []
            count = 0
            max_issues = 20 # Limit results
            for issue in issues:
                if count >= max_issues: break
                issue_list.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "user": issue.user.login,
                    "url": issue.html_url,
                    "created_at": issue.created_at.isoformat()
                })
                count += 1
            logger.info(f"Fetched {len(issue_list)} issues for {owner}/{repo} (state={state})")
            return {"status": "success", "content": json.dumps(issue_list, indent=2)}

        # --- Get Issue Details ---
        elif action == "get_issue":
            issue_number = params.get("issue_number")
            if not owner or not repo or not issue_number: return {"status": "error", "message": "Action 'get_issue' requires 'owner', 'repo', and 'issue_number'."}
            repo_obj = gh_client.get_repo(f"{owner}/{repo}")
            try:
                issue = repo_obj.get_issue(issue_number)
                comments_pag = issue.get_comments()
                comments_list = []
                comment_count = 0
                max_comments = 10
                for comment in comments_pag:
                     if comment_count >= max_comments: break
                     comments_list.append({
                         "user": comment.user.login,
                         "created_at": comment.created_at.isoformat(),
                         "body": comment.body[:500] + ("..." if len(comment.body) > 500 else "") # Truncate long comments
                     })
                     comment_count+=1

                content = {
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "user": issue.user.login,
                    "body": issue.body[:5000] + ("..." if len(issue.body) > 5000 else ""), # Truncate long body
                    "url": issue.html_url,
                    "created_at": issue.created_at.isoformat(),
                    "comments_count": issue.comments,
                    "comments_preview": comments_list
                }
                logger.info(f"Fetched issue details for {owner}/{repo}#{issue_number}")
                return {"status": "success", "content": json.dumps(content, indent=2)}
            except UnknownObjectException:
                logger.error(f"Issue not found: {owner}/{repo}#{issue_number}")
                return {"status": "error", "message": f"Issue #{issue_number} not found."}

        # --- Search Code ---
        elif action == "search_code":
             query = params.get("query")
             if not query: return {"status": "error", "message": "Action 'search_code' requires a 'query'."}
             # Add repo qualifier if owner/repo provided
             if owner and repo:
                 query += f" repo:{owner}/{repo}"

             results = gh_client.search_code(query) # Paginated
             code_list = []
             count = 0
             max_results = 10 # Limit search results
             for item in results:
                  if count >= max_results: break
                  # Get snippet if available (text_matches is sometimes empty)
                  snippet = f"Path: {item.path}" # Fallback
                  if item.text_matches:
                      snippet = item.text_matches[0].fragment # Use first match fragment

                  code_list.append({
                      "name": item.name,
                      "path": item.path,
                      "sha": item.sha,
                      "url": item.html_url,
                      "repo": item.repository.full_name,
                      "snippet": snippet[:500] + ("..." if len(snippet)>500 else "") # Truncate
                  })
                  count += 1
             logger.info(f"GitHub code search for '{query}' returned {len(code_list)} results.")
             return {"status": "success", "content": json.dumps(code_list, indent=2)}

        else:
            return {"status": "error", "message": f"Unsupported GitHub action: {action}"}

    except GithubException as e:
        logger.error(f"GitHub API error for action '{action}': {e.status} - {e.data}")
        return {"status": "error", "message": f"GitHub API error: {e.status} {e.data.get('message', '')}"}
    except Exception as e:
        logger.error(f"Unexpected error during GitHub inspection (action: {action}): {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}


def get_wayback_machine_snapshot(url: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetches the content of a webpage snapshot from the Wayback Machine (archive.org)
    using WaybackMachineCDXServerAPI for better reliability.

    Args:
        url (str): The URL of the webpage to look up.
        timestamp (Optional[str]): The desired timestamp in YYYYMMDDhhmmss format (e.g., "20230601120000").
                                   If None, fetches the most recent available snapshot.

    Returns:
        dict: Dictionary with 'status', 'content' (snapshot text content, cleaned by fetch_webpage_content),
              'snapshot_url', 'actual_timestamp', or 'message' if an error occurred.
    """
    logger.info(f"Fetching Wayback Machine snapshot for URL: {url}, Timestamp: {timestamp} using CDXServerAPI")
    if not WAYBACKPY_AVAILABLE:
        logger.warning("Waybackpy library not available. Cannot fetch snapshot.")
        return {"status": "error", "message": "Waybackpy library not available."}

    try:
        # 使用 WaybackMachineCDXServerAPI
        cdx_api = WaybackMachineCDXServerAPI(url, USER_AGENT)
        snapshot_object = None  # 这将持有 waybackpy 的 Snapshot 对象

        if timestamp:
            try:
                dt_object = datetime.datetime.strptime(timestamp, '%Y%m%d%H%M%S')
                logger.info(f"Attempting to find snapshot near {dt_object.isoformat()} using CDX.")
                # waybackpy 3.0.6 的 near 方法接受年、月、日、时、分
                snapshot_object = cdx_api.near(
                    year=dt_object.year,
                    month=dt_object.month,
                    day=dt_object.day,
                    hour=dt_object.hour,
                    minute=dt_object.minute
                )
            except ValueError as ve:  # 来自 strptime 的错误
                logger.error(f"Invalid timestamp format provided: {timestamp}. Error: {ve}")
                return {"status": "error", "message": f"Invalid timestamp format: {timestamp}. Use YYYYMMDDhhmmss."}
            except TypeError as te:  # 万一 near() 的参数仍有问题
                logger.error(f"TypeError calling cdx_api.near for {url} with ts {timestamp}: {te}", exc_info=True)
                return {"status": "error",
                        "message": f"Wayback Machine CDX library error (TypeError in .near()): {str(te)}"}
        else:
            logger.info("Timestamp not provided, fetching the newest snapshot using CDX.")
            snapshot_object = cdx_api.newest()

        if not snapshot_object or not hasattr(snapshot_object, 'archive_url') or not snapshot_object.archive_url:
            logger.warning(
                f"No Wayback Machine snapshot object obtained or it lacks an archive_url (from CDX) for {url}, timestamp: {timestamp or 'newest'}.")
            # NoCDXRecordFound 应该在下面被捕获，这里是针对返回了非预期对象的情况
            return {"status": "error",
                    "message": f"No valid Wayback Machine snapshot found (from CDX) for {url} (timestamp: {timestamp or 'newest'})."}

        logger.info(f"Fetching content from snapshot URL: {snapshot_object.archive_url}")
        # 调用您项目中实际的 fetch_webpage_content 函数
        fetched_data = fetch_webpage_content(snapshot_object.archive_url,
                                             use_readability=True)  # 假设您的函数也接受 use_readability

        if fetched_data["status"] == "success":
            logger.info(f"Successfully fetched and processed Wayback Machine snapshot (from CDX).")
            actual_ts_str = "N/A"
            # Snapshot 对象 (来自 CDXServerAPI 的 .near() 或 .newest()) 应该有 datetime_timestamp 属性
            if hasattr(snapshot_object, 'datetime_timestamp') and isinstance(snapshot_object.datetime_timestamp,
                                                                             datetime.datetime):
                actual_ts_str = snapshot_object.datetime_timestamp.isoformat()

            return {
                "status": "success",
                "content": fetched_data["content"],  # 由您的 fetch_webpage_content 清理和截断
                "snapshot_url": snapshot_object.archive_url,
                "actual_timestamp": actual_ts_str
            }
        else:
            logger.error(
                f"Failed to fetch content from snapshot URL {snapshot_object.archive_url}: {fetched_data['message']}")
            return {"status": "error",
                    "message": f"Found snapshot URL (from CDX) but failed to fetch its content: {fetched_data['message']}"}

    except requests.exceptions.ConnectionError as e:  # 网络连接错误 (可能在 fetch_webpage_content 中发生)
        logger.error(f"Connection error for {url} or its snapshot: {e}")
        return {"status": "error", "message": f"Connection error: {str(e)}"}
    except NoCDXRecordFound:
        logger.warning(f"NoCDXRecordFound for {url} with timestamp {timestamp}")
        return {"status": "error", "message": f"No snapshots found for URL via CDX: {url}"}
    except WaybackError as wbe:  # 其他 waybackpy 特定的错误
        logger.error(f"A Waybackpy specific error occurred for {url} (CDX): {wbe}", exc_info=True)
        return {"status": "error", "message": f"Wayback Machine CDX library error: {str(wbe)}"}
    except AttributeError as ae:
        # 例如，如果 snapshot_object 为 None，尝试访问 .archive_url
        logger.error(f"Attribute error during Wayback Machine CDX operation for {url}: {ae}", exc_info=True)
        return {"status": "error",
                "message": f"Wayback Machine internal library error (AttributeError with CDX): {str(ae)}"}
    except Exception as e:
        logger.error(f"Generic error fetching Wayback Machine snapshot for {url} (CDX): {e}", exc_info=True)
        return {"status": "error", "message": f"Wayback Machine fetch failed (Generic with CDX): {str(e)}"}