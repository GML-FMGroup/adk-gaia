# src/agents/__init__.py
from .orchestrator import orchestrator_agent
from .google_search_agent import google_search_agent
from .builtin_code_executor_agent import builtin_code_executor_agent
from .web_researcher import web_researcher_agent
from .code_executor import code_executor_agent
from .document_processor import document_processor_agent
from .spreadsheet_data_agent import spreadsheet_data_agent # 确保导入路径正确
from .multimodal_processor import multimodal_processor_agent
from .specialized_file_agent import specialized_file_agent
from .calculator_logic_agent import calculator_logic_agent
from .filesystem_agent import filesystem_agent

root_agent = orchestrator_agent

__all__ = [
    "orchestrator_agent",
    "google_search_agent",
    "builtin_code_executor_agent",
    "web_researcher_agent",
    "code_executor_agent",
    "document_processor_agent",
    "spreadsheet_data_agent", # 确保导出
    "multimodal_processor_agent",
    "specialized_file_agent",
    "calculator_logic_agent",
    "filesystem_agent",
    "root_agent"
]