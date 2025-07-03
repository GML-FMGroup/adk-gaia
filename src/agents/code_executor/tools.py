# src/tools/code_tools.py
import subprocess
import sys
import tempfile
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 获取项目根目录 (假设此文件在 src/tools/ 下)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
logger.info(f"Project root directory identified as: {PROJECT_ROOT}")

def execute_local_python_code(code: str) -> Dict[str, Any]:
    """
    Executes a given Python code string locally and securely within the project's working directory.
    Uses the same Python interpreter that the agent is running with.

    Args:
        code (str): The Python code string to execute.

    Returns:
        dict: A dictionary containing the execution status, stdout, and stderr.
              Example success: {"status": "success", "stdout": "Hello World!", "stderr": ""}
              Example error: {"status": "error", "stdout": "", "stderr": "SyntaxError: invalid syntax"}
    """
    logger.info(f"Attempting to execute local Python code snippet (first 100 chars): {code[:100]}...")

    # 使用临时文件来执行代码，更安全一些
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False, dir=PROJECT_ROOT, encoding='utf-8') as temp_script:
            temp_script.write(code)
            temp_script_path = temp_script.name

        # 使用与当前环境相同的 Python 解释器执行脚本
        # 在项目根目录下执行，允许相对路径导入等
        process = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,  # Set working directory to project root
            timeout=30 # 添加超时限制
        )

        os.remove(temp_script_path) # 执行后删除临时文件

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode == 0:
            logger.info(f"Local Python code executed successfully. Stdout: {stdout[:200]}...")
            return {"status": "success", "stdout": stdout, "stderr": stderr}
        else:
            logger.error(f"Local Python code execution failed. Stderr: {stderr}")
            return {"status": "error", "stdout": stdout, "stderr": stderr, "returncode": process.returncode}

    except subprocess.TimeoutExpired:
        logger.error("Local Python code execution timed out.")
        # 尝试再次删除文件，以防万一
        try:
            if 'temp_script_path' in locals() and os.path.exists(temp_script_path):
                 os.remove(temp_script_path)
        except Exception:
            pass # 忽略删除错误
        return {"status": "error", "message": "Code execution timed out after 30 seconds."}
    except Exception as e:
        logger.error(f"Unexpected error executing local Python code: {e}", exc_info=True)
        # 尝试再次删除文件
        try:
            if 'temp_script_path' in locals() and os.path.exists(temp_script_path):
                 os.remove(temp_script_path)
        except Exception:
            pass
        return {"status": "error", "message": f"An unexpected error occurred: {str(e)}"}