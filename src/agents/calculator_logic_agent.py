# src/agents/calculator_logic_agent.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging

# 导入我们定义的计算工具函数
from src.tools.calculation_tools import (
    evaluate_mathematical_expression,
    calculate_statistics,
    unit_converter,
    calculate_checksum,
    newtons_method_solver
)

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

CALCULATOR_LOGIC_MODEL = get_model("specialist_model_flash") # Flash 模型通常足够

if not CALCULATOR_LOGIC_MODEL:
    raise ValueError("Model for CalculatorLogicAgent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
evaluate_expression_tool = FunctionTool(func=evaluate_mathematical_expression)
calculate_statistics_tool = FunctionTool(func=calculate_statistics)
unit_converter_tool = FunctionTool(func=unit_converter)
calculate_checksum_tool = FunctionTool(func=calculate_checksum)
newtons_method_tool = FunctionTool(func=newtons_method_solver)

calculator_logic_agent = LlmAgent(
    name="CalculatorLogicAgent",
    model=CALCULATOR_LOGIC_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门执行数学和逻辑运算。接收一个包含任务描述和参数的 'request' 字符串。"
        "能够调用的工具包括：\n"
        "- `evaluate_mathematical_expression`: 安全地评估数学表达式字符串。需要参数: `expression` (str)。\n"
        "- `calculate_statistics`: 计算数值列表的统计数据。需要参数: `data` (List[float]), `stat_types` (List[str], e.g., ['mean', 'stdev'])。\n"
        "- `unit_converter`: 在不同单位之间转换数值。需要参数: `value` (float), `original_unit` (str), `target_unit` (str)。\n"
        "- `calculate_checksum`: 计算数字序列的校验和。需要参数: `number_sequence` (str), `algorithm` (str, e.g., 'isbn10', 默认为 'isbn10')。\n"
        "- `newtons_method_solver`: 使用牛顿法求解方程的根。需要参数: `function_str` (str, e.g., 'x**2 - 4'), `initial_guess` (float), 可选 `derivative_str` (str), `tolerance` (float), `max_iterations` (int)。"
        "协调器应确保 'request' 字符串清晰地包含要执行的操作类型以及所有必需的参数。"
    ),
    instruction=(
        "你是一个专业的计算和逻辑智能体。你会收到一个名为 `request` 的字符串参数，"
        "其中包含关于特定计算或逻辑任务的指令。\n"
        "**重要：** 你的任务是解析 `request` 字符串以提取必要的参数，然后调用最合适的工具。\n"
        "1.  **解析请求：** 仔细阅读输入的 `request` 字符串。\n"
        "    - 识别所需的**计算类型**（例如，求值表达式、计算统计数据、单位转换、校验和、牛顿法）。\n"
        "    - 从请求中提取该计算所需的**所有必要参数**。例如：\n"
        "        - 对于 `evaluate_mathematical_expression`：`expression` 字符串。\n"
        "        - 对于 `calculate_statistics`：一个 `data` 列表和一个 `stat_types` 列表。\n"
        "        - 对于 `unit_converter`：`value`、`original_unit` 和 `target_unit`。\n"
        "        - 对于 `calculate_checksum`：`number_sequence` 以及可选的 `algorithm`（默认为 'isbn10'）。\n"
        "        - 对于 `newtons_method_solver`：`function_str`、`initial_guess` 以及可选的 `derivative_str`、`tolerance`、`max_iterations`。\n"
        "2.  **选择工具：** 根据识别出的计算类型选择正确的工具。\n"
        "3.  **执行工具：** 使用提取的参数调用所选工具。\n"
        "    - 确保用于 `calculate_statistics` 的数值列表已正确格式化为 Python 数字列表。\n"
        "    - 对于 `evaluate_mathematical_expression`，确保表达式是该工具的有效字符串。\n"
        "4.  **返回结果：** 从工具的输出字典中返回 'result'、'results'、'checksum_digit'、'root' 或 'message'。准确地传达任何错误消息。"
        "    如果工具返回一个包含 'status': 'success' 和一个结果字段（例如 'result'、'results'、'checksum_digit'、'root'）的字典，"
        "    则返回该结果字段的值。如果状态是 'error'，则返回 'message' 字段。"
    ),
    tools=[
        evaluate_expression_tool,
        calculate_statistics_tool,
        unit_converter_tool,
        calculate_checksum_tool,
        newtons_method_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"CalculatorLogicAgent initialized with model: {CALCULATOR_LOGIC_MODEL}")
logger.info(f"CalculatorLogicAgent Tools: {[tool.name for tool in calculator_logic_agent.tools]}")