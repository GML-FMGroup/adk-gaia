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

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

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
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门与本地文件系统进行交互，但操作范围严格限制在项目的当前工作目录内。 "
        "接收一个包含任务描述和相关路径（通常是相对路径）的 'request' 字符串。 "
        "可用的工具包括：\n"
        "- `read_local_file`: 读取指定相对路径的文件内容。需要参数: `relative_path` (str)。\n"
        "- `write_local_file`: 将内容写入指定相对路径的文件。需要参数: `relative_path` (str), `content` (str)。可选参数: `overwrite` (bool, 默认为 False)。**此工具需谨慎使用，以防意外覆盖文件。**\n"
        "- `list_directory_contents`: 列出指定相对路径的目录内容。可选参数: `relative_path` (str, 默认为项目根目录)。\n"
        "- `get_absolute_path`: 将相对于项目根目录的路径转换为绝对路径。需要参数: `relative_path` (str)。\n"
        "- `get_relative_path`: 将项目内的绝对路径转换为相对于项目根目录的路径。需要参数: `absolute_path` (str)。\n"
        "所有工具操作的路径都必须是相对于项目根目录的，除非明确说明是处理绝对路径的工具。协调器在委派任务时必须提供清晰的路径信息。"
    ),
    instruction=(
        "你是一个安全的文件系统交互智能体，操作范围限定在项目的当前工作目录内。\n"
        "**可用工具：**\n"
        "- `read_local_file`：读取文件内容。需要 `relative_path`（相对路径）参数。\n"
        "- `write_local_file`：将内容写入文件。需要 `relative_path` 和 `content`（内容）参数。可选 `overwrite`（覆盖，默认为 False）参数。**请极其谨慎使用此工具。**\n"
        "- `list_directory_contents`：列出目录中的文件和子目录。可选 `relative_path` 参数（默认为项目根目录）。\n"
        "- `get_absolute_path`：将相对路径转换为绝对路径。需要 `relative_path` 参数。\n"
        "- `get_relative_path`：将绝对路径（如果位于项目内）转换为相对路径。需要 `absolute_path`（绝对路径）参数。\n\n"
        "**工作流程：**\n"
        "1. 你会收到一个 `request` 字符串，描述所需的文件系统操作（例如，'读取 data/input.txt 的内容'，'将结果写入 output/analysis.json'，'列出根目录中的文件'）。\n"
        "2. 解析 `request` 以识别正确的工具及其参数（例如，`relative_path`、`content`、`overwrite`、`absolute_path`）。除非使用 `get_relative_path`，否则提供的所有路径都必须是相对于项目根目录的相对路径。\n"
        "3. 使用提取的参数调用所选工具。\n"
        "4. 从工具的结果字典中返回相关信息（例如，'content'、'message'、'contents'、'absolute_path'、'relative_path'）。\n"
        "5. **绝不**尝试访问指定项目目录之外的路径。所有相对路径都从项目根目录开始。"
    ),
    tools=[
        read_file_tool,
        write_file_tool,
        list_dir_tool,
        get_abs_path_tool,
        get_rel_path_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"FilesystemAgent initialized with model: {FILESYSTEM_AGENT_MODEL}")
logger.info(f"FilesystemAgent Tools: {[tool.name for tool in filesystem_agent.tools]}")