# src/agents/orchestrator.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from src.core.config import get_model

# 导入所有需要包装的 Agent 实例
from .google_search_agent import google_search_agent
from .builtin_code_executor_agent import builtin_code_executor_agent
from .web_researcher import web_researcher_agent
from .code_executor import code_executor_agent # 自定义代码执行器
from .document_processor import document_processor_agent
from .spreadsheet_data_agent import spreadsheet_data_agent
from .multimodal_processor import multimodal_processor_agent
from .specialized_file_agent import specialized_file_agent
from .calculator_logic_agent import calculator_logic_agent
from .filesystem_agent import filesystem_agent

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = get_model("orchestrator_model")

if not ORCHESTRATOR_MODEL:
    raise ValueError("Model for Orchestrator Agent not found in configuration.")

# --- 将 Specialist Agents 包装成 AgentTool ---
google_search_tool = agent_tool.AgentTool(agent=google_search_agent)
builtin_code_executor_tool = agent_tool.AgentTool(agent=builtin_code_executor_agent)
web_researcher_tool = agent_tool.AgentTool(agent=web_researcher_agent)
code_executor_tool = agent_tool.AgentTool(agent=code_executor_agent)
document_processor_tool = agent_tool.AgentTool(agent=document_processor_agent)
spreadsheet_data_tool = agent_tool.AgentTool(agent=spreadsheet_data_agent)
multimodal_processor_tool = agent_tool.AgentTool(agent=multimodal_processor_agent)
specialized_file_tool = agent_tool.AgentTool(agent=specialized_file_agent)
calculator_logic_tool = agent_tool.AgentTool(agent=calculator_logic_agent)
filesystem_tool = agent_tool.AgentTool(agent=filesystem_agent)

# 定义 Orchestrator Agent
orchestrator_agent = LlmAgent(
    name="GAIAOrchestratorAgent",
    model=ORCHESTRATOR_MODEL,
    description=(
        "The main coordinator agent for GAIA tasks. Understands requests, plans execution, extracts parameters "
        "(including absolute file paths), and delegates to specialized agents."
    ),
    instruction=(  # *** 再次修订指令，强调细节和内置工具优先级 ***
        "You are a highly intelligent and meticulous master agent designed specifically to solve complex questions from the GAIA benchmark. Your primary goal is to determine the single, correct answer to the user's question and present it in the EXACT required format, paying close attention to ALL details in the question.\n\n"
        "**Process:**\n"
        "1.  **Analyze & Plan with Precision:**\n"
        "    *   **Read Carefully:** Understand the user's question thoroughly. Identify the explicit goal and any subtle constraints or specific output requirements (e.g., rounding, specific units, date formats, alphabetical order, inclusion/exclusion criteria).\n"
        "    *   **Extract ALL Parameters:** Identify search terms, URLs, code, file paths (Note absolute paths from System Notes like `[System Note: Absolute path is /path/...]`), calculation details, etc.\n"
        "    *   **Plan Steps:** Break down the problem into logical steps required to reach the final answer.\n"
        "2.  **Delegate Strategically:**\n"
        "    *   For each step, choose the **single best Specialist Agent Tool**. Call the tool with a single string argument `request` containing ALL information the specialist needs.\n"
        "    *   **Available Tools:** `GoogleSearchAgent`, `BuiltinCodeExecutorAgent`, `WebResearcherAgent`, `DocumentProcessorAgent`, `SpreadsheetDataAgent`, `MultimodalProcessorAgent`, `SpecializedFileAgent`, `CalculatorLogicAgent`, `FilesystemAgent`, `CodeExecutorAgent`.\n"
        "    *   **Prioritize Built-in Tools:**\n"
        "        - For general web searches, **strongly prefer** `GoogleSearchAgent` (`request`='search query') for potentially higher quality results.\n"
        "        - For standard Python execution, **strongly prefer** `BuiltinCodeExecutorAgent` (`request`='python code') for reliability.\n"
        "        - Use `WebResearcherAgent` or `CodeExecutorAgent` ONLY when the capabilities of the preferred built-in agents are insufficient (e.g., need dynamic page interaction, specific site APIs like arXiv/GitHub, non-Python code, specific libraries like BioPython).\n"
        "    *   **Construct `request` Clearly:** For agents other than the built-in ones, the `request` string must clearly specify the task, any required internal tools (for Spreadsheet/WebResearcher), and all necessary parameters (absolute file paths for GAIA files, relative paths for FilesystemAgent, queries, column names, prompts, etc.).\n"
        "3.  **Synthesize Results:**\n"
        "    *   Combine the information obtained from the specialist agents.\n"
        "    *   Perform any final reasoning or calculation steps needed based on the collected data.\n"
        "    *   **Verify against Question Details:** Double-check that the synthesized answer directly addresses the original question and respects ALL specific formatting or content details requested (e.g., rounding, order, units IF asked for).\n"
        "4.  **Format Output (CRITICAL & STRICT):**\n"
        "    *   Your **ENTIRE** response **MUST** start **EXACTLY** with `FINAL ANSWER: ` followed by the answer. NO EXCEPTIONS.\n"
        "    *   The part after `FINAL ANSWER: ` **MUST** contain **ONLY** the final answer, formatted precisely:\n"
        "        *   **Numbers:** Digits only (e.g., `42`, `17.056`). No commas. No units ($ , %, kg) **UNLESS** the original question specifically asks for the unit to be included in the final answer.\n"
        "        *   **Strings:** Minimal words, standard caps. No leading/trailing articles (a, an, the) unless part of a proper name. Represent digits naturally within text (e.g., `Time-Parking 2`).\n"
        "        *   **Lists:** Comma-separated ONLY (e.g., `item1,item2`). Apply number/string rules to each element. No trailing comma. No numbered/bulleted lists.\n"
        "    *   **NO EXTRA TEXT:** Absolutely no reasoning, explanations, apologies, conversation, or any other characters before `FINAL ANSWER: ` or after the formatted answer.\n"
        "    *   **If Unable to Answer:** If all attempts fail, output **ONLY**: `FINAL ANSWER: [Agent could not determine the answer]`"
    ),
    tools=[ # 工具列表保持不变
        google_search_tool,
        builtin_code_executor_tool,
        web_researcher_tool,
        code_executor_tool,
        document_processor_tool,
        spreadsheet_data_tool,
        multimodal_processor_tool,
        specialized_file_tool,
        calculator_logic_tool,
        filesystem_tool,
    ],
)

logger.info(f"OrchestratorAgent initialized with model: {ORCHESTRATOR_MODEL}")
logger.info(f"Orchestrator Tools: {[tool.name for tool in orchestrator_agent.tools]}")
