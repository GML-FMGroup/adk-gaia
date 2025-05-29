# src/agents/specialized_file_agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging

# 导入相关的工具函数
from src.tools.file_tools import (
    parse_pdb_file,
    extract_zip_content,
    read_json_file # JSON/JSONL/JSONLD 也可以被认为是特殊文本文件
)

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

SPECIALIZED_FILE_AGENT_MODEL = get_model("specialist_model_flash")

if not SPECIALIZED_FILE_AGENT_MODEL:
    raise ValueError("Model for Specialized File Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
parse_pdb_tool = FunctionTool(func=parse_pdb_file)
extract_zip_tool = FunctionTool(func=extract_zip_content)
read_json_tool = FunctionTool(func=read_json_file) # 重用 JSON 工具

specialized_file_agent = LlmAgent(
    name="SpecializedFileAgent",
    model=SPECIALIZED_FILE_AGENT_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门处理特定的文件格式，包括 PDB (Protein Data Bank), ZIP 压缩包, 以及结构化的 JSON, JSONL, JSONLD 文件。"
        "接收一个包含操作指令和文件绝对路径的 'request' 字符串。"
        "根据文件类型和指令选择合适的工具：\n"
        "- `parse_pdb_file`: 解析 PDB 文件并返回结构摘要或回答关于PDB文件的特定查询。需要参数: `file_path` (str, PDB文件的绝对路径)。工具的 prompt/query 参数用于传递特定查询（例如 '计算原子间距离'）。\n"
        "- `extract_zip_content`: 列出 ZIP 压缩包的内容，或提取压缩包内特定文件的文本内容。需要参数: `file_path` (str, ZIP文件的绝对路径)。可选参数: `target_filename` (str, 要提取的内部文件名)。如果未提供 `target_filename`，则列出内容。\n"
        "- `read_json_file`: 读取并解析 JSON, JSONL, 或 JSONLD 文件，返回其内容。需要参数: `file_path` (str, 文件的绝对路径)。\n"
        "协调器应确保 'request' 字符串中包含清晰的操作指令（如对PDB的特定查询、ZIP中要提取的文件名）和正确的绝对文件路径。"
    ),
    instruction=(
        "你是一位精通特殊文件格式的专家。你会收到一个名为 `request` 的字符串参数，"
        "其中包含指令和一个绝对文件路径。\n"
        "**重要：** 你的任务是解析 `request` 字符串以提取文件路径和具体操作，然后调用相应的工具。\n"
        "1.  **解析请求：** 从输入的 `request` 字符串中提取**绝对文件路径**和**操作**（例如，'解析此 PDB 文件并总结'，'列出此 zip 文件的内容'，'从此 zip 文件中提取 specific_file.txt'，'读取此 jsonld 文件'）。对于 ZIP 文件，如果指定了 `target_filename`（例如，'使用 target_filename specific_file.txt'），也需提取。对于 PDB 文件，如果提到了任何特定查询（例如，'计算距离...'），也需提取。\n"
        "2.  **根据文件扩展名选择工具：**\n"
        "    - 对于 `.pdb` 文件，使用 `parse_pdb_file`。将任何提取的查询作为 `query` 参数传递。\n"
        "    - 对于 `.zip` 文件，使用 `extract_zip_content`。传递任何提取的 `target_filename`。\n"
        "    - 对于 `.json`、`.jsonl`、`.jsonld` 文件，使用 `read_json_file`。\n"
        "3.  **执行工具：** 调用所选工具，将提取的**文件路径**作为 `file_path` 参数传递，并传递任何其他相关参数（`target_filename`、`query`）。\n"
        "4.  **返回结果：** 传递工具输出中的 'content' 或 'message'。"
    ),
    tools=[
        parse_pdb_tool,
        extract_zip_tool,
        read_json_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"SpecializedFileAgent initialized with model: {SPECIALIZED_FILE_AGENT_MODEL}")
logger.info(f"SpecializedFileAgent Tools: {[tool.name for tool in specialized_file_agent.tools]}")