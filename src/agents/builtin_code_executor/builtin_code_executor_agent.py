# src/agents/builtin_code_executor_agent.py
import logging
from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
from src.core.config import get_model

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

BUILTIN_CODE_EXECUTOR_MODEL = get_model("specialist_model_flash")

if not BUILTIN_CODE_EXECUTOR_MODEL:
    raise ValueError("Model for BuiltinCodeExecutorAgent not found in configuration.")

builtin_code_executor_agent = LlmAgent(
    name="BuiltinCodeExecutorAgent",
    model=BUILTIN_CODE_EXECUTOR_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门使用 ADK 内置的 `BuiltInCodeExecutor` 在云端沙箱中执行 Python 代码片段。 "
        "此智能体用于标准的 Python 计算或简单的脚本执行，不能访问本地文件系统。 "
        "期望的输入是一个包含待执行 Python 代码的字符串，智能体将返回代码的 stdout 或 stderr。"
    ),
    instruction=(
        "你是一个专门的 Python 代码执行智能体。你唯一的功能就是使用 `built_in_code_execution` 工具。"
        "这是一个位于云端沙箱的Python代码执行环境，所以请不要尝试用 `built_in_code_execution` 访问任何本地的内容。"
        "你会收到一个名为 `code` 的字符串参数，其中包含要执行的 Python 代码。"
        "使用 `built_in_code_execution` 工具执行该代码。"
        "返回代码执行生成的标准输出 (stdout) 或错误消息。"
    ),
    code_executor=BuiltInCodeExecutor(),  # Only the built-in tool
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback,
)

logger.info(
    f"BuiltinCodeExecutorAgent initialized with model: {BUILTIN_CODE_EXECUTOR_MODEL}"
)
logger.info(f"BuiltinCodeExecutorAgent configured with BuiltInCodeExecutor.")
