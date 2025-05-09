# src/tools/filesystem_tools.py
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# 获取项目根目录 (假设此文件在 src/tools/ 下)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
logger.info(f"Filesystem tools operating within project root: {PROJECT_ROOT}")

def _get_safe_abs_path(relative_path_str: str) -> Optional[Path]:
    """
    Constructs an absolute path from a relative path within the project root
    and verifies it does not escape the project root directory.

    Args:
        relative_path_str: The relative path string from the project root.

    Returns:
        A Path object representing the absolute path if safe, otherwise None.
    """
    try:
        # 清理路径，移除 ".." 等
        # 注意：os.path.normpath 不能完全防止路径遍历，需要配合 startswith 检查
        normalized_relative = os.path.normpath(relative_path_str).replace('\\', '/')
        if normalized_relative.startswith(('../', '/')):
             logger.warning(f"Potentially unsafe relative path detected: {relative_path_str}")
             return None

        abs_path = Path(PROJECT_ROOT) / normalized_relative
        # 使用 resolve() 来处理符号链接并获取最终的绝对路径
        resolved_path = abs_path.resolve()

        # 检查解析后的路径是否仍在项目根目录下
        if str(resolved_path).startswith(str(PROJECT_ROOT)):
            return resolved_path
        else:
            logger.warning(f"Path escape attempt detected: '{relative_path_str}' resolved to '{resolved_path}', which is outside project root '{PROJECT_ROOT}'")
            return None
    except Exception as e:
        logger.error(f"Error resolving path '{relative_path_str}': {e}")
        return None


def read_local_file(relative_path: str) -> Dict[str, Any]:
    """
    Reads the content of a file located relative to the project's root directory.

    Args:
        relative_path (str): The path to the file, relative to the project root.

    Returns:
        dict: Status and content or error message.
              Example: {"status": "success", "content": "File content here..."}
    """
    logger.info(f"Attempting to read local file: {relative_path}")
    safe_path = _get_safe_abs_path(relative_path)

    if not safe_path:
        return {"status": "error", "message": f"Access denied or invalid path: {relative_path}"}

    try:
        if not safe_path.is_file():
            logger.error(f"Path is not a file or does not exist: {safe_path}")
            return {"status": "error", "message": f"File not found or is not a file: {relative_path}"}

        # 尝试以 UTF-8 读取，如果失败，尝试 latin-1
        try:
            content = safe_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Could not decode {relative_path} as UTF-8, trying latin-1.")
            content = safe_path.read_text(encoding='latin-1', errors='replace')

        logger.info(f"Successfully read local file: {relative_path}")
        # Truncate long content
        max_len = 10000
        if len(content) > max_len:
            logger.warning(f"Local file content truncated: {relative_path}")
            content = content[:max_len] + "\n... (truncated)"
        return {"status": "success", "content": content}

    except FileNotFoundError: # 理论上 is_file() 会先捕捉，但保留以防万一
        logger.error(f"File not found error for: {safe_path}")
        return {"status": "error", "message": f"File not found: {relative_path}"}
    except Exception as e:
        logger.error(f"Error reading local file {relative_path}: {e}", exc_info=True)
        return {"status": "error", "message": f"Error reading file: {str(e)}"}

def write_local_file(relative_path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    """
    Writes content to a file located relative to the project's root directory.
    **Warning:** Use with extreme caution. Allows overwriting existing files if overwrite=True.

    Args:
        relative_path (str): The path to the file, relative to the project root.
        content (str): The content to write to the file.
        overwrite (bool): If True, overwrite the file if it exists. Defaults to False.

    Returns:
        dict: Status and success/error message.
              Example: {"status": "success", "message": "Successfully wrote to data/output.txt"}
    """
    logger.warning(f"Attempting to write local file: {relative_path} (Overwrite: {overwrite}) - Use with caution!")
    safe_path = _get_safe_abs_path(relative_path)

    if not safe_path:
        return {"status": "error", "message": f"Access denied or invalid path: {relative_path}"}

    # 额外的安全检查：不允许写入项目根目录下的特定敏感文件或目录
    forbidden_patterns = ['.git', '.env', 'pyproject.toml', 'config.json', 'src/'] # 示例
    for pattern in forbidden_patterns:
         # 检查解析后的绝对路径是否以禁止的模式开头（需要更精确的匹配）
         # 简化：直接比较相对路径是否匹配或在其下
         normalized_relative = os.path.normpath(relative_path).replace('\\', '/')
         if normalized_relative == pattern or normalized_relative.startswith(pattern + '/'):
              logger.error(f"Attempted to write to a forbidden path: {relative_path}")
              return {"status": "error", "message": f"Writing to '{relative_path}' is not allowed."}

    try:
        if safe_path.exists() and not overwrite:
            logger.warning(f"File already exists and overwrite is False: {safe_path}")
            return {"status": "error", "message": f"File already exists: {relative_path}. Set overwrite=True to replace."}

        if safe_path.is_dir():
             logger.error(f"Cannot write file, path exists and is a directory: {safe_path}")
             return {"status": "error", "message": f"Path exists and is a directory: {relative_path}"}

        # 创建父目录
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        safe_path.write_text(content, encoding='utf-8')

        logger.info(f"Successfully wrote to local file: {relative_path}")
        return {"status": "success", "message": f"Successfully wrote to {relative_path}"}

    except Exception as e:
        logger.error(f"Error writing local file {relative_path}: {e}", exc_info=True)
        return {"status": "error", "message": f"Error writing file: {str(e)}"}

def list_directory_contents(relative_path: str = ".") -> Dict[str, Any]:
    """
    Lists the contents (files and subdirectories) of a directory relative to the project root.

    Args:
        relative_path (str): The path to the directory, relative to the project root. Defaults to "." (project root).

    Returns:
        dict: Status and a list of item names, or an error message.
              Example: {"status": "success", "contents": ["file1.txt", "subdir1", "image.png"]}
    """
    logger.info(f"Attempting to list directory contents: {relative_path}")
    safe_path = _get_safe_abs_path(relative_path)

    if not safe_path:
        return {"status": "error", "message": f"Access denied or invalid path: {relative_path}"}

    try:
        if not safe_path.is_dir():
            logger.error(f"Path is not a directory or does not exist: {safe_path}")
            return {"status": "error", "message": f"Directory not found or is not a directory: {relative_path}"}

        contents = [item.name for item in safe_path.iterdir()]
        logger.info(f"Successfully listed contents for: {relative_path}")
        return {"status": "success", "contents": contents}

    except FileNotFoundError: # 理论上 is_dir() 会先捕捉
        logger.error(f"Directory not found error for: {safe_path}")
        return {"status": "error", "message": f"Directory not found: {relative_path}"}
    except Exception as e:
        logger.error(f"Error listing directory {relative_path}: {e}", exc_info=True)
        return {"status": "error", "message": f"Error listing directory: {str(e)}"}

def get_absolute_path(relative_path: str) -> Dict[str, Any]:
    """
    Returns the absolute path for a given relative path within the project directory.

    Args:
        relative_path (str): The path relative to the project root.

    Returns:
        dict: Status and the absolute path or an error message.
    """
    logger.info(f"Getting absolute path for: {relative_path}")
    safe_path = _get_safe_abs_path(relative_path)
    if safe_path:
         return {"status": "success", "absolute_path": str(safe_path)}
    else:
         return {"status": "error", "message": f"Invalid or unsafe relative path: {relative_path}"}

def get_relative_path(absolute_path: str) -> Dict[str, Any]:
    """
    Returns the relative path from the project root for a given absolute path,
    if the absolute path is within the project directory.

    Args:
        absolute_path (str): The absolute path.

    Returns:
        dict: Status and the relative path or an error message.
    """
    logger.info(f"Getting relative path for: {absolute_path}")
    try:
        resolved_abs_path = Path(absolute_path).resolve()
        if str(resolved_abs_path).startswith(str(PROJECT_ROOT)):
             relative_p = resolved_abs_path.relative_to(PROJECT_ROOT)
             return {"status": "success", "relative_path": str(relative_p)}
        else:
             logger.warning(f"Absolute path {absolute_path} is outside project root {PROJECT_ROOT}")
             return {"status": "error", "message": "Path is outside the project directory."}
    except Exception as e:
         logger.error(f"Error calculating relative path for {absolute_path}: {e}")
         return {"status": "error", "message": f"Could not determine relative path: {str(e)}"}