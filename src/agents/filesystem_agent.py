# src/agents/filesystem_agent.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model

# 导入文件系统工具函数
from src.tools.filesystem_tools import (
    read_local_file,
    write_local_file,
    list_directory_contents,
    get_absolute_path,
    get_relative_path,
)

logger = logging.getLogger(__name__)

FILESYSTEM_AGENT_MODEL = get_model("specialist_model_flash")

if not FILESYSTEM_AGENT_MODEL:
    raise ValueError("Model for FilesystemAgent not found in configuration.")

# --- 包装文件系统工具 ---
read_file_tool = FunctionTool(func=read_local_file)
write_file_tool = FunctionTool(func=write_local_file)
list_dir_tool = FunctionTool(func=list_directory_contents)
get_abs_path_tool = FunctionTool(func=get_absolute_path)
get_rel_path_tool = FunctionTool(func=get_relative_path)

filesystem_agent = LlmAgent(
    name="FilesystemAgent",
    model=FILESYSTEM_AGENT_MODEL,
    description=(
        "Specializes in interacting with the local filesystem *within the project's working directory*. "
        "Can read files, write files (use cautiously!), list directory contents, and resolve paths."
    ),
    instruction=(
        "You are a secure filesystem interaction agent, limited to the project's working directory.\n"
        "**Available Tools:**\n"
        "- `read_local_file`: Reads content from a file. Requires `relative_path`.\n"
        "- `write_local_file`: Writes content to a file. Requires `relative_path` and `content`. Optional `overwrite` (default False). **Use with EXTREME CAUTION.**\n"
        "- `list_directory_contents`: Lists files and subdirectories. Optional `relative_path` (defaults to project root).\n"
        "- `get_absolute_path`: Converts a relative path to its absolute path. Requires `relative_path`.\n"
        "- `get_relative_path`: Converts an absolute path (if within project) to a relative path. Requires `absolute_path`.\n\n"
        "**Workflow:**\n"
        "1. You will receive a `request` string describing the filesystem operation needed (e.g., 'Read the contents of data/input.txt', 'Write the result to output/analysis.json', 'List files in the root directory').\n"
        "2. Parse the `request` to identify the correct tool and its parameters (e.g., `relative_path`, `content`, `overwrite`, `absolute_path`). All paths provided must be relative to the project root unless using `get_relative_path`.\n"
        "3. Call the chosen tool with the extracted parameters.\n"
        "4. Return the relevant information from the tool's result dictionary (e.g., 'content', 'message', 'contents', 'absolute_path', 'relative_path').\n"
        "5. **NEVER** attempt to access paths outside the designated project directory. All relative paths start from the project root."
    ),
    tools=[
        read_file_tool,
        write_file_tool,
        list_dir_tool,
        get_abs_path_tool,
        get_rel_path_tool,
    ],
)

logger.info(f"FilesystemAgent initialized with model: {FILESYSTEM_AGENT_MODEL}")
logger.info(f"FilesystemAgent Tools: {[tool.name for tool in filesystem_agent.tools]}")