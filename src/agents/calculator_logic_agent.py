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
    description=(
        "Specializes in performing mathematical calculations, statistical analysis, unit conversions, "
        "checksum calculations (like ISBN), and solving numerical methods like Newton's method. "
        "It takes specific parameters for each operation."
    ),
    instruction=(
        "You are an expert calculation and logic agent. You will receive a single string argument named `request` "
        "containing instructions about a specific calculation or logical task.\n"
        "**IMPORTANT:** Your task is to parse the `request` string to extract the necessary parameters and then call the MOST appropriate tool.\n"
        "1.  **Parse Request:** Carefully read the input `request` string.\n"
        "    - Identify the **type of calculation** needed (e.g., evaluate expression, calculate stats, convert units, checksum, Newton's method).\n"
        "    - Extract all **necessary arguments** for that calculation from the request. For example:\n"
        "        - For `evaluate_mathematical_expression`: the `expression` string.\n"
        "        - For `calculate_statistics`: a list of `data` and a list of `stat_types`.\n"
        "        - For `unit_converter`: the `value`, `original_unit`, and `target_unit`.\n"
        "        - For `calculate_checksum`: the `number_sequence` and optionally the `algorithm` (defaults to 'isbn10').\n"
        "        - For `newtons_method_solver`: the `function_str`, `initial_guess`, and optionally `derivative_str`, `tolerance`, `max_iterations`.\n"
        "2.  **Select Tool:** Choose the correct tool based on the identified calculation type.\n"
        "3.  **Execute Tool:** Call the selected tool with the extracted arguments.\n"
        "    - Ensure numerical lists for `calculate_statistics` are correctly formatted as Python lists of numbers.\n"
        "    - For `evaluate_mathematical_expression`, ensure the expression is a valid string for the tool.\n"
        "4.  **Return Result:** Return the 'result', 'results', 'checksum_digit', 'root', or 'message' from the tool's output dictionary. Relay any error messages accurately."
        "    If the tool returns a dictionary with 'status': 'success' and a result field (e.g. 'result', 'results', 'checksum_digit', 'root'), "
        "    return the value of that result field. If the status is 'error', return the 'message' field."
    ),
    tools=[
        evaluate_expression_tool,
        calculate_statistics_tool,
        unit_converter_tool,
        calculate_checksum_tool,
        newtons_method_tool,
    ],
)

logger.info(f"CalculatorLogicAgent initialized with model: {CALCULATOR_LOGIC_MODEL}")
logger.info(f"CalculatorLogicAgent Tools: {[tool.name for tool in calculator_logic_agent.tools]}")