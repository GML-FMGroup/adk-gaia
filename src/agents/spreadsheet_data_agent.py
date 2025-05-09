# src/agents/spreadsheet_data_agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging
from typing import Union, Optional # 导入 Union, Optional

# 导入所有相关的工具函数
from src.tools.spreadsheet_tools import (
    get_spreadsheet_info,
    get_sheet_names,
    get_cell_value,
    query_spreadsheet,
    calculate_column_stat,
    # read_spreadsheet # We can replace this with more specific tools or keep it as a fallback
)

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
    description=( # 更新描述
        "Specializes in reading, analyzing, and querying data from spreadsheet files "
        "like Excel (.xlsx) and CSV (.csv). Can get sheet info, sheet names, cell values, "
        "filter data with queries, and calculate column statistics."
    ),
    instruction=( # *** 大幅更新指令以反映新工具 ***
        "You are an expert spreadsheet data analyst. You will receive a single string argument named `request` "
        "containing instructions and an absolute file path to a spreadsheet.\n"
        "**IMPORTANT:** Parse the `request` string to determine the specific task and required parameters, then call the MOST appropriate tool.\n"
        "**Available Tools:**\n"
        "- `get_spreadsheet_info`: Provides metadata (shape, columns, types, stats, sample rows). Requires `file_path`, optional `sheet_name`.\n"
        "- `get_sheet_names`: Lists all sheet names in an Excel file. Requires `file_path`.\n"
        "- `get_cell_value`: Gets the value of a single cell. Requires `file_path`, `cell_coordinate` (e.g., 'B5'), optional `sheet_name`.\n"
        "- `query_spreadsheet`: Filters data using a pandas query string. Requires `file_path`, `query_string`, optional `sheet_name`.\n"
        "- `calculate_column_stat`: Calculates statistics for a column. Requires `file_path`, `column_name`, `stat_type` (e.g., 'sum', 'mean', 'std'), optional `sheet_name`.\n\n"
        "**Workflow:**\n"
        "1.  **Parse Request:** Extract the **absolute file path** and the **specific action** (get info, get sheets, get cell, query, calculate stat) from the `request`. Also extract all necessary parameters for that action (e.g., `sheet_name`, `cell_coordinate`, `query_string`, `column_name`, `stat_type`).\n"
        "2.  **Select & Execute Tool:** Call the chosen tool with the correctly named arguments based on the parsed request.\n"
        "3.  **Return Result:** Relay the relevant information from the tool's output dictionary (e.g., 'info', 'sheet_names', 'value', 'filtered_data', 'result'). If the status is 'error', return the 'message'."
    ),
    tools=[ # 列出所有新工具
        get_info_tool,
        get_sheets_tool,
        get_cell_tool,
        query_tool,
        calculate_stat_tool,
        # read_spreadsheet_tool, # Optional fallback
    ],
)

logger.info(f"SpreadsheetDataAgent initialized with model: {SPREADSHEET_AGENT_MODEL}")
logger.info(f"SpreadsheetDataAgent Tools: {[tool.name for tool in spreadsheet_data_agent.tools]}")