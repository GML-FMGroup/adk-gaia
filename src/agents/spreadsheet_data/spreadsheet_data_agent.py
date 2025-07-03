# src/agents/spreadsheet_data_agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging
from typing import Union, Optional # 导入 Union, Optional

# 导入所有相关的工具函数
from src.agents.spreadsheet_data.tools import (
    get_spreadsheet_info,
    get_sheet_names,
    get_cell_value,
    query_spreadsheet,
    calculate_column_stat,
    # read_spreadsheet # We can replace this with more specific tools or keep it as a fallback
)

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

SPREADSHEET_AGENT_MODEL = get_model("specialist_model_flash") # Flash might still be okay

if not SPREADSHEET_AGENT_MODEL:
    raise ValueError("Model for Spreadsheet Data Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
get_info_tool = FunctionTool(func=get_spreadsheet_info)
get_sheets_tool = FunctionTool(func=get_sheet_names)
get_cell_tool = FunctionTool(func=get_cell_value)
query_tool = FunctionTool(func=query_spreadsheet)
calculate_stat_tool = FunctionTool(func=calculate_column_stat)
# Optional: Keep read_spreadsheet as a fallback or remove if covered by get_info + query
# read_spreadsheet_tool = FunctionTool(func=read_spreadsheet)


spreadsheet_data_agent = LlmAgent(
    name="SpreadsheetDataAgent",
    model=SPREADSHEET_AGENT_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=( # 更新描述
        "专门用于读取、分析和查询电子表格文件（如 Excel .xlsx, .xls 和 CSV .csv）中的数据。 "
        "接收一个包含任务指令和电子表格绝对文件路径的 'request' 字符串。 "
        "根据指令解析并调用以下工具之一：\n"
        "- `get_spreadsheet_info`: 提供电子表格的元数据（如形状、列名、数据类型、基本统计、样本行）。需要参数: `file_path` (str, 绝对路径)。可选参数: `sheet_name` (str, 工作表名称或索引)。\n"
        "- `get_sheet_names`: 列出 Excel 文件中的所有工作表名称。需要参数: `file_path` (str, 绝对路径)。仅适用于 Excel 文件。\n"
        "- `get_cell_value`: 获取指定单元格的值。需要参数: `file_path` (str, 绝对路径), `cell_coordinate` (str, 例如 'B5' 或 'Sheet1!B5')。可选参数: `sheet_name` (str, 如果未在 cell_coordinate 中指定)。\n"
        "- `query_spreadsheet`: 使用 Pandas 查询字符串筛选电子表格中的数据。需要参数: `file_path` (str, 绝对路径), `query_string` (str, 例如 \"`列名 A` > 10 and `列名 B` == 'some_value'\")。可选参数: `sheet_name` (str)。\n"
        "- `calculate_column_stat`: 计算指定列的统计数据。需要参数: `file_path` (str, 绝对路径), `column_name` (str), `stat_type` (str, 例如 'sum', 'mean', 'median', 'std', 'count')。可选参数: `sheet_name` (str)。\n"
        "协调器应确保 'request' 字符串清晰地指明要执行的操作以及所有必需的参数，包括正确的绝对文件路径和任何工作表特定信息。"
    ),
    instruction=(
        "你是一位专业的电子表格数据分析师。你会收到一个名为 `request` 的字符串参数，"
        "其中包含指令和一个指向电子表格的绝对文件路径。\n"
        "**重要：** 解析 `request` 字符串以确定具体任务和所需参数，然后调用最合适的工具。\n"
        "**可用工具：**\n"
        "- `get_spreadsheet_info`：提供元数据（形状、列、类型、统计信息、样本行）。需要 `file_path`，可选 `sheet_name`。\n"
        "- `get_sheet_names`：列出 Excel 文件中的所有工作表名称。需要 `file_path`。\n"
        "- `get_cell_value`：获取单个单元格的值。需要 `file_path`、`cell_coordinate`，可选 `sheet_name`。\n"
        "- `query_spreadsheet`：使用 pandas 查询字符串筛选数据。需要 `file_path`、`query_string`，可选 `sheet_name`。\n"
        "- `calculate_column_stat`：计算列的统计数据。需要 `file_path`、`column_name`、`stat_type`（例如，'sum'、'mean'、'std'），可选 `sheet_name`。\n\n"
        "**工作流程：**\n"
        "1.  **解析请求：** 从 `request` 中提取**绝对文件路径**和**具体操作**（获取信息、获取工作表、获取单元格、查询、计算统计数据）。同时提取该操作所需的所有必要参数（例如，`sheet_name`、`cell_coordinate`、`query_string`、`column_name`、`stat_type`）。\n"
        "2.  **选择并执行工具：** 根据解析的请求，使用正确命名的参数调用所选工具。\n"
        "3.  **返回结果：** 传递工具输出字典中的相关信息（例如，'info'、'sheet_names'、'value'、'filtered_data'、'result'）。如果状态为 'error'，则返回 'message'。"
    ),
    tools=[ # 列出所有新工具
        get_info_tool,
        get_sheets_tool,
        get_cell_tool,
        query_tool,
        calculate_stat_tool,
        # read_spreadsheet_tool, # Optional fallback
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"SpreadsheetDataAgent initialized with model: {SPREADSHEET_AGENT_MODEL}")
logger.info(f"SpreadsheetDataAgent Tools: {[tool.name for tool in spreadsheet_data_agent.tools]}")