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
    description=(
        "Handles specialized file formats such as PDB (Protein Data Bank), "
        "ZIP archives, and structured JSON/JSONL/JSONLD files. Can parse PDB, list or extract ZIP contents, "
        "and read JSON-based files."
    ),
    instruction=(
        "You are an expert in specialized file formats. You will receive a single string argument named `request` "
        "containing instructions and an absolute file path.\n"
        "**IMPORTANT:** Your task is to parse the `request` string to extract the file path and the specific action, then call the appropriate tool.\n"
        "1.  **Parse Request:** Extract the **absolute file path** and the **action** (e.g., 'parse this PDB file and summarize', 'list contents of this zip', 'extract specific_file.txt from this zip', 'read this jsonld file') from the input `request` string. For ZIP files, also extract the `target_filename` if specified (e.g., 'using target_filename specific_file.txt'). For PDB files, extract any specific query if mentioned (e.g. 'calculate distance...').\n"
        "2.  **Select Tool based on file extension:**\n"
        "    - For `.pdb` files, use `parse_pdb_file`. Pass any extracted query as the `query` argument.\n"
        "    - For `.zip` files, use `extract_zip_content`. Pass any extracted `target_filename`.\n"
        "    - For `.json`, `.jsonl`, `.jsonld` files, use `read_json_file`.\n"
        "3.  **Execute Tool:** Call the selected tool, passing the extracted **file path** as the `file_path` argument, and any other relevant arguments (`target_filename`, `query`).\n"
        "4.  **Return Result:** Relay the 'content' or 'message' from the tool's output."
    ),
    tools=[
        parse_pdb_tool,
        extract_zip_tool,
        read_json_tool,
    ],
)

logger.info(f"SpecializedFileAgent initialized with model: {SPECIALIZED_FILE_AGENT_MODEL}")
logger.info(f"SpecializedFileAgent Tools: {[tool.name for tool in specialized_file_agent.tools]}")