# src/agents/code_executor.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model

# 导入新的本地代码执行工具
from src.tools.code_tools import execute_local_python_code
# 导入未来可能需要的自定义代码工具
# from src.tools.code_tools import run_biopython_script, etc.

logger = logging.getLogger(__name__)

CODE_EXECUTOR_MODEL = get_model("specialist_model_flash")

if not CODE_EXECUTOR_MODEL:
    raise ValueError("Model for CodeExecutorAgent not found in configuration.")

# --- 包装本地 Python 执行工具 ---
execute_local_python_tool = FunctionTool(func=execute_local_python_code)

code_executor_agent = LlmAgent(
    name="CodeExecutorAgent",
    model=CODE_EXECUTOR_MODEL,
    description=( # 更新描述
        "Specializes in executing provided Python code snippets locally and securely within the project's working directory. "
        "Can also handle future custom execution tools (e.g., BioPython, other languages)."
    ),
    instruction=( # 更新指令
        "You are a specialized code execution agent. Your primary capability is executing Python code locally using the `execute_local_python_code` tool.\n"
        "**Workflow:**\n"
        "1. You will receive a `request` string containing the Python code to execute.\n"
        "2. Use the `execute_local_python_code` tool, passing the code string as the `code` argument.\n"
        "3. Return the standard output (`stdout`) and standard error (`stderr`) from the execution result dictionary provided by the tool.\n"
        "4. If the tool indicates an error status, relay the error message or stderr.\n"
        "5. (Future capabilities might involve other tools for specific languages or libraries.)\n"
        "6. **DO NOT** use the built-in code executor. Only use the provided `execute_local_python_code` tool."
    ),
    tools=[
        execute_local_python_tool,
        # Add other custom code tools here in the future
    ]
)

logger.info(f"CodeExecutorAgent initialized with model: {CODE_EXECUTOR_MODEL}")
logger.info(f"CodeExecutorAgent Tools: {[tool.name for tool in code_executor_agent.tools]}")