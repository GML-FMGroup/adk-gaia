# src/tools/calculation_tools.py
import math
import statistics
import re
from typing import Dict, Any, List, Optional, Union # Union 将不再用于 data 参数
import logging

# 使用安全的表达式求值库，如 numexpr 或 sympy.sympify
# 避免直接使用 eval()
try:
    import numexpr
    def safe_eval_expression(expression: str) -> Any:
        # 移除非法字符，只允许数字、基本运算符、括号、点、科学计数法e/E
        # 以及 numexpr 支持的函数 (如果需要，可以预定义允许的函数列表)
        # 注意：numexpr 本身有自己的安全机制，但多一层过滤总是好的
        # 这是一个非常基础的白名单，实际应用中可能需要更复杂的策略
        # 移除非法字符，只允许数字、基本运算符、括号、点、科学计数法e/E
        # 以及一些数学函数（如果numexpr直接支持）
        # 确保只包含数字、运算符、括号、点、科学计数法e/E以及已知的安全函数
        # 这是一个简化的白名单，实际中可能需要更复杂的策略或使用更安全的库
        allowed_chars = r"[0-9\.\+\-\*\/\(\)\sdegasincoanqrlpmxbtwhE]" # 增加了 math 函数可能用到的字母
        # 检查是否有不允许的字符（非字母数字且不在白名单内的特殊字符）
        # 或者是否有潜在的恶意代码模式
        if not re.match(r"^[a-zA-Z0-9\s\.\+\-\*\/\(\)\^eE%]*$", expression): # 允许了^和%
            # 进一步检查，不允许连续的字母组合除非它们是已知函数的一部分
            # 这是一个复杂的问题，简单的正则表达式难以完美解决。
            # 此处我们依赖 numexpr 的安全性，但仍发出警告
            logging.warning(f"Expression '{expression}' contains potentially unsafe characters/patterns. Proceeding with numexpr.")


        # 定义一个numexpr能识别的数学函数字典
        # 注意：numexpr默认支持很多numpy函数，但明确指定更安全
        local_dict = {
            'pi': math.pi,
            'e': math.e,
            'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
            'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
            'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
            'abs': abs, 'min': min, 'max': max, # Python 内置函数
            # 如果需要，可以添加更多 numexpr 支持的函数
        }
        # NumExpr 计算，限制可用的函数
        return numexpr.evaluate(expression, local_dict=local_dict).item()

except ImportError:
    logging.warning("numexpr not installed. Falling back to sympy for safe_eval_expression. `pip install numexpr` is recommended for performance.")
    try:
        from sympy import sympify, N
        def safe_eval_expression(expression: str) -> Any:
            # Sympy 的 sympify 更安全，但可能较慢
            # 它会将表达式转换为符号表达式再求值
            # 对于纯数值计算，确保结果是数值类型
            try:
                result = sympify(expression)
                # 如果结果是符号，尝试数值化
                if not result.is_Number:
                    # 尝试使用 N 进行数值评估，如果失败，则可能表达式不适合直接数值化
                    eval_result = N(result)
                    if eval_result.is_Number:
                        return float(eval_result) if eval_result.is_Float else int(eval_result)
                    else: # 如果仍然不是数字，可能是一个复杂的符号表达式
                        raise ValueError("Expression did not evaluate to a simple number.")
                return float(result) if result.is_Float else int(result)

            except (SyntaxError, TypeError, ValueError) as e:
                logging.error(f"Sympy evaluation error for '{expression}': {e}")
                raise ValueError(f"Invalid or non-numeric expression for sympy: {expression}")

    except ImportError:
        logging.error("Sympy not installed. Mathematical expression evaluation will be unsafe or fail. Please install numexpr or sympy.")
        def safe_eval_expression(expression: str) -> Any:
            raise NotImplementedError("No safe evaluation library (numexpr or sympy) is available.")


try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    PINT_AVAILABLE = True
    logging.info("Pint library found for unit conversion.")
except ImportError:
    PINT_AVAILABLE = False
    logging.warning("Pint library not installed. Unit conversion tool will not be available. `pip install pint`")
    ureg = None


logger = logging.getLogger(__name__)

def evaluate_mathematical_expression(expression: str) -> Dict[str, Any]:
    """
    Evaluates a given mathematical expression string securely and returns the result.
    Supports basic arithmetic (+, -, *, /), exponentiation (^ or **), parentheses,
    and common math functions like sqrt, sin, cos, tan, log, log10, abs.
    Args:
        expression (str): The mathematical expression to evaluate (e.g., "(5 + 7) * 3", "sqrt(16) + sin(0.5)").
    Returns:
        dict: A dictionary with 'status' and 'result' or 'message'.
              Example: {"status": "success", "result": 36}
                       {"status": "error", "message": "Invalid expression"}
    """
    logger.info(f"Attempting to evaluate mathematical expression: {expression}")
    try:
        # 预处理：将常见的数学函数名转为小写，确保兼容性
        # expression = expression.lower() # 可能导致问题，如果变量名大小写敏感
        # 将'^' 替换为 '**' 以兼容python和numexpr/sympy
        expression = expression.replace('^', '**')
        result = safe_eval_expression(expression)
        logger.info(f"Successfully evaluated expression '{expression}': {result}")
        return {"status": "success", "result": result}
    except (ValueError, SyntaxError, TypeError, NameError, ZeroDivisionError, OverflowError, NotImplementedError) as e:
        logger.error(f"Error evaluating expression '{expression}': {e}")
        return {"status": "error", "message": f"Error evaluating expression: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error evaluating expression '{expression}': {e}", exc_info=True)
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}

def calculate_statistics(data: List[float], stat_types: List[str]) -> Dict[str, Any]: # MODIFIED HERE
    """
    Calculates specified statistical measures for a list of numbers.
    Supported stat_types: "mean", "median", "mode", "stdev" (sample standard deviation),
                         "pstdev" (population standard deviation), "variance", "pvariance".
    Args:
        data (List[float]): A list of numerical data. Integers will be implicitly converted.
        stat_types (List[str]): A list of strings specifying which statistics to calculate.
                                Example: ["mean", "stdev"]
    Returns:
        dict: A dictionary with 'status' and 'results' (a dict of calculated stats) or 'message'.
              Example: {"status": "success", "results": {"mean": 10.5, "stdev": 2.1}}
    """
    logger.info(f"Attempting to calculate statistics for data (first 5: {data[:5]}...) with types: {stat_types}")
    if not isinstance(data, list) or not all(isinstance(x, (int, float)) for x in data): # 内部仍检查 int 和 float
        logger.error("Invalid data format for statistics: not a list of numbers.")
        return {"status": "error", "message": "Input data must be a list of numbers."}
    if not data:
        logger.warning("Empty data list provided for statistics.")
        return {"status": "error", "message": "Cannot calculate statistics on empty data."}

    # 将所有数据转换为 float 以进行统计计算，确保一致性
    float_data = [float(x) for x in data]

    results = {}
    available_stats = {
        "mean": statistics.mean,
        "median": statistics.median,
        "mode": statistics.mode,
        "stdev": statistics.stdev,
        "pstdev": statistics.pstdev,
        "variance": statistics.variance,
        "pvariance": statistics.pvariance,
    }

    for stat_type in stat_types:
        stat_type_lower = stat_type.lower()
        if stat_type_lower in available_stats:
            try:
                # Mode can raise StatisticsError if no unique mode or data is empty
                # Stdev/variance require at least 2 data points for sample versions
                if (stat_type_lower in ["stdev", "variance"] and len(float_data) < 2):
                    results[stat_type] = "N/A (requires at least 2 data points)"
                    logger.warning(f"Cannot calculate {stat_type} for data with length {len(float_data)}.")
                    continue
                results[stat_type] = available_stats[stat_type_lower](float_data)
            except statistics.StatisticsError as se:
                logger.warning(f"StatisticsError for {stat_type} on data: {se}")
                results[stat_type] = f"N/A ({str(se)})"
            except Exception as e:
                logger.error(f"Error calculating {stat_type}: {e}")
                results[stat_type] = f"Error ({str(e)})"
        else:
            logger.warning(f"Unsupported statistic type: {stat_type}")
            results[stat_type] = "Unsupported statistic type"

    if not results:
        return {"status": "error", "message": "No valid statistics were calculated."}

    logger.info(f"Successfully calculated statistics: {results}")
    return {"status": "success", "results": results}


def unit_converter(value: float, original_unit: str, target_unit: str) -> Dict[str, Any]:
    """
    Converts a value from an original unit to a target unit using the Pint library.
    Args:
        value (float): The numerical value to convert.
        original_unit (str): The original unit (e.g., "meter", "kg", "mph", "celsius").
        target_unit (str): The target unit (e.g., "kilometer", "pound", "kph", "fahrenheit").
    Returns:
        dict: A dictionary with 'status' and 'result' (converted value with unit) or 'message'.
              Example: {"status": "success", "result": "1.609 km"}
    """
    logger.info(f"Attempting to convert {value} {original_unit} to {target_unit}")
    if not PINT_AVAILABLE:
        message = "Unit conversion tool is unavailable because 'pint' library is not installed."
        logger.error(message)
        return {"status": "error", "message": message}

    try:
        original_quantity = ureg.Quantity(value, original_unit)
        converted_quantity = original_quantity.to(target_unit)
        # 返回带单位的字符串，或者数值和单位分开
        # result_str = f"{converted_quantity.magnitude:.4f} {converted_quantity.units}"
        result_val = round(converted_quantity.magnitude, 6) # 保留一些精度
        logger.info(f"Successfully converted: {value} {original_unit} -> {result_val} {converted_quantity.units}")
        return {"status": "success", "result_value": result_val, "result_unit": str(converted_quantity.units)}
    except Exception as e: # Pint can raise various errors (UndefinedUnitError, DimensionalityError)
        logger.error(f"Error converting units from {original_unit} to {target_unit}: {e}")
        return {"status": "error", "message": f"Unit conversion error: {str(e)}"}

def calculate_checksum(number_sequence: str, algorithm: str = "isbn10") -> Dict[str, Any]:
    """
    Calculates a checksum digit for a given number sequence based on a specified algorithm.
    Currently supports "isbn10" and "isbn13_custom".
    For "isbn13_custom", it expects an additional 'weights' parameter in the request
    if the default (1,3 alternating) is not used.
    Args:
        number_sequence (str): The sequence of digits (as a string, hyphens will be removed).
        algorithm (str): The checksum algorithm to use. Defaults to "isbn10".
                         Supported: "isbn10", "isbn13_custom".
                         For "isbn13_custom", the problem might specify custom weights.
                         This tool's LLM interface should be instructed on how to pass custom weights if needed.
                         For GAIA, the prompt to CalculatorLogicAgent will need to specify this.
    Returns:
        dict: A dictionary with 'status' and 'checksum_digit' or 'message'.
    """
    logger.info(f"Calculating checksum for '{number_sequence}' using algorithm '{algorithm}'")
    digits_str = number_sequence.replace('-', '').replace(' ', '')

    if not digits_str.isdigit():
        return {"status": "error", "message": "Input sequence must contain only digits after removing hyphens/spaces."}

    if algorithm.lower() == "isbn10":
        if len(digits_str) != 9: # ISBN-10 check digit is for the first 9 digits
            return {"status": "error", "message": "ISBN-10 algorithm requires 9 digits to calculate the 10th check digit."}
        s = 0
        for i, digit in enumerate(digits_str):
            s += int(digit) * (10 - i)
        check_digit_val = (11 - (s % 11)) % 11
        check_digit = 'X' if check_digit_val == 10 else str(check_digit_val)
        logger.info(f"ISBN-10 checksum for {digits_str}: {check_digit}")
        return {"status": "success", "checksum_digit": check_digit}

    elif algorithm.lower() == "isbn13_custom":
        # GAIA's task d56db2318-640f-477a-a82f-bc93ad13e882 implies a specific weight for isbn13-like numbers
        # This tool needs to be called with the correct 'digits_str' (12 digits for standard ISBN-13 check)
        # and the 'custom_weight_factor' (e.g., 7 as per that GAIA task).
        # The Orchestrator would need to parse this from the GAIA question.
        # This tool itself is generic; the LLM calling it must provide the correct params.
        # The question asks: "checksum digit is calculated with an alternate weight of 1 and some other positive integer less than 10"
        # So, one weight is 1, the other is 'custom_weight_factor'.
        # Let's assume the `number_sequence` here is the 12 digits for which we calculate the 13th.
        # And the `custom_weight_factor` is passed by the LLM (this example doesn't take it as direct arg,
        # but CalculatorLogicAgent's LLM should be prompted to extract it and pass it if this tool were to be called by it)
        # For simplicity here, let's assume a fixed custom weight for a demonstration,
        # or the LLM calling CalculatorLogicAgent would need to provide it in the `expression` or a dedicated param.
        #
        # **Simplification for this direct tool:**
        # The GAIA task is very specific. This tool function would ideally take `custom_weight_factor` as an argument.
        # Since `FunctionTool` doesn't easily allow dynamic arguments based on `algorithm`,
        # the LLM (CalculatorLogicAgent) would need to be instructed to format the `expression`
        # for `evaluate_mathematical_expression` to include this logic, or we'd need a more specific tool.
        #
        # Let's make this tool require the 12 digits and the *single* custom_weight_factor.
        # The Orchestrator/CalculatorLogicAgent must extract this custom_weight_factor from the problem.
        # This function will expect `number_sequence` to be the 12 digits.
        # The `custom_weight_factor` should ideally be part of the call from the CalculatorLogicAgent.
        # For now, this tool will just return an error if algorithm is 'isbn13_custom' without more context.
        # A better approach: CalculatorLogicAgent should parse the GAIA question, identify it needs
        # this specific custom ISBN logic, and then use `evaluate_mathematical_expression` by
        # constructing the summation formula directly if the custom weight is known.
        logger.warning("Algorithm 'isbn13_custom' is complex and usually problem-specific. "
                       "It's better handled by CodeExecutorAgent or by evaluate_mathematical_expression "
                       "if the formula can be constructed by the LLM.")
        return {"status": "error", "message": "Algorithm 'isbn13_custom' should be handled by constructing the sum expression for 'evaluate_mathematical_expression' or by CodeExecutorAgent for problem-specific logic."}

    else:
        return {"status": "error", "message": f"Unsupported checksum algorithm: {algorithm}"}


def newtons_method_solver(function_str: str, initial_guess: float, derivative_str: Optional[str] = None, tolerance: float = 1e-7, max_iterations: int = 100) -> Dict[str, Any]:
    """
    Finds a root of a function using Newton's method.
    The function and its derivative (if provided) must be evaluatable by `safe_eval_expression`.
    Args:
        function_str (str): The function f(x) as a string (e.g., "x**3 + 4*x**2 - 3*x + 8"). 'x' is the variable.
        initial_guess (float): The initial guess for the root.
        derivative_str (Optional[str]): The derivative f'(x) as a string. If None, numerical differentiation might be attempted (not implemented here for simplicity, Sympy could do it).
                                       For GAIA, it's likely the derivative will be calculable or given.
        tolerance (float): The desired precision for the root.
        max_iterations (int): Maximum number of iterations to perform.
    Returns:
        dict: Status and the root, or an error message.
    """
    logger.info(f"Newton's method for f(x)='{function_str}', x0={initial_guess}, f'(x)='{derivative_str}'")

    # 定义内部函数用于安全的求值，避免在循环中重复替换和编译表达式
    # 这假设 safe_eval_expression 可以处理 'x' 作为变量的表达式
    def eval_f(val, expr_str):
        return safe_eval_expression(expr_str.replace('x', f"({val})"))

    # 初始化 f_prime_str
    f_prime_str_to_use = derivative_str

    if not derivative_str:
        try:
            from sympy import sympify, diff, Symbol, lambdify
            x_sym = Symbol('x')
            f_sym = sympify(function_str)
            df_sym = diff(f_sym, x_sym)
            f_prime_str_to_use = str(df_sym) # 将符号导数转为字符串
            logger.info(f"Calculated derivative using sympy: f'(x) = {f_prime_str_to_use}")
        except ImportError:
             return {"status": "error", "message": "Derivative string (f'(x)) must be provided if sympy is not available."}
        except Exception as e_sympy:
            logger.error(f"Error with sympy for derivative calculation: {e_sympy}")
            return {"status": "error", "message": f"Error calculating derivative with sympy: {str(e_sympy)}"}

    if not f_prime_str_to_use: # 再次检查，如果sympy失败且没有提供初始derivative_str
        return {"status": "error", "message": "Derivative could not be determined."}

    x_n = initial_guess
    for i in range(max_iterations):
        try:
            f_xn = eval_f(x_n, function_str)
            df_xn = eval_f(x_n, f_prime_str_to_use)

            if abs(df_xn) < 1e-10:
                logger.error("Derivative is too small; Newton's method fails.")
                return {"status": "error", "message": "Derivative too small, method fails."}

            x_n_plus_1 = x_n - f_xn / df_xn

            if abs(x_n_plus_1 - x_n) < tolerance:
                logger.info(f"Newton's method converged after {i+1} iterations to root: {x_n_plus_1}")
                return {"status": "success", "root": round(x_n_plus_1, 7), "iterations": i + 1}
            x_n = x_n_plus_1
        except (ValueError, TypeError, NameError, ZeroDivisionError, OverflowError) as e:
            logger.error(f"Error during Newton's method iteration {i+1} for x_n={x_n}: {e}")
            return {"status": "error", "message": f"Error in iteration {i+1}: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during Newton's method iteration {i+1}: {e}", exc_info=True)
            return {"status": "error", "message": f"An unexpected error occurred in iteration {i+1}: {str(e)}"}

    logger.warning("Newton's method did not converge within max iterations.")
    return {"status": "error", "message": "Method did not converge within max iterations."}