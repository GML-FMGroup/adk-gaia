# src/agents/code_executor.py
import logging
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model

# 导入新的本地代码执行工具
from src.agents.code_executor.tools import execute_local_python_code
# 导入未来可能需要的自定义代码工具
# from src.tools.code_tools import run_biopython_script, etc.

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

CODE_EXECUTOR_MODEL = get_model("specialist_model_flash")

if not CODE_EXECUTOR_MODEL:
    raise ValueError("Model for CodeExecutorAgent not found in configuration.")

# --- 包装本地 Python 执行工具 ---
execute_local_python_tool = FunctionTool(func=execute_local_python_code)

code_executor_agent = LlmAgent(
    name="CodeExecutorAgent",
    model=CODE_EXECUTOR_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=( # 更新描述
        "专门在项目的工作目录内本地且安全地执行 Python 代码片段。 "
        "与 `BuiltinCodeExecutorAgent` 不同，此智能体执行的代码可以与本地项目环境交互（例如，读写项目内的文件，如果代码本身包含此类逻辑）。"
        "主要工具是 `execute_local_python_code`，它需要一个名为 `code` (str) 的参数，包含要执行的 Python 代码。"
        "返回代码执行的 stdout 和 stderr。"
        "适用于需要本地环境交互或 `BuiltinCodeExecutorAgent` 功能不足的 Python 代码执行场景。"
    ),
    instruction=( # 更新指令
        "你是一个专门的代码执行智能体。你的主要能力是使用 `execute_local_python_code` 工具在本地执行 Python 代码。\n"
        "**工作流程：**\n"
        "1. 你会收到一个包含要执行的 Python 代码的 `request` 字符串。\n"
        "2. 使用 `execute_local_python_code` 工具，将代码字符串作为 `code` 参数传递。\n"
        "3. 从工具提供的执行结果字典中返回标准输出 (`stdout`) 和标准错误 (`stderr`)。\n"
        "4. 如果工具指示错误状态，则传达错误消息或 stderr。\n"
    ),
    tools=[
        execute_local_python_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"CodeExecutorAgent initialized with model: {CODE_EXECUTOR_MODEL}")
logger.info(f"CodeExecutorAgent Tools: {[tool.name for tool in code_executor_agent.tools]}")