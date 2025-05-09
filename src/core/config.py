# src/core/config.py
import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# 确保 CONFIG_PATH 正确指向项目根目录下的 config.json
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(project_root, 'config.json')

def load_config() -> Dict[str, Any]:
    """Loads the configuration from config.json."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        logger.info(f"Configuration loaded successfully from {CONFIG_PATH}")
        return config_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {CONFIG_PATH}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading config: {e}")

# 加载一次配置供全局使用
try:
    APP_CONFIG = load_config()
except (FileNotFoundError, ValueError, RuntimeError) as e:
    logger.error(f"Error loading application configuration: {e}")
    # 提供一个默认的最小配置，以防文件加载失败
    APP_CONFIG = {
        "orchestrator_model": "gemini-2.5-pro-preview-05-06", # 使用更新后的模型
        "specialist_model_flash": "gemini-2.5-flash-preview-04-17", # 使用更新后的模型
        "specialist_model_pro": "gemini-2.5-pro-preview-05-06", # 使用更新后的模型
        "ollama_model": None,
        "gaia_data_dir": "./GAIA/2023",
        "api_port": 9012,
        "runner_strategy": "all",
        "runner_task_id": None,
        "runner_first_n": None,
        "runner_max_retries": 0,
        "runner_max_workers": 1
    }
    logger.warning("Warning: Using default fallback configuration.")

# --- 修改 get_model 函数 ---
def get_model(model_key: str) -> Optional[str]:
    """Safely retrieves a model name directly from the loaded configuration."""
    # 直接从 APP_CONFIG 获取，不再查找嵌套的 "models" 字典
    model_name = APP_CONFIG.get(model_key)
    if model_name is None:
        logger.warning(f"Model key '{model_key}' not found in config.json.")
    return model_name
# --- 结束修改 ---

def get_gaia_data_dir() -> Optional[str]:
    """Retrieves the GAIA data directory path and ensures it's absolute."""
    rel_path = APP_CONFIG.get("gaia_data_dir")
    if rel_path:
        # project_root 在文件顶部已定义
        return os.path.abspath(os.path.join(project_root, rel_path))
    logger.warning("gaia_data_dir not found in config.json.")
    return None

def get_api_port() -> int:
    """Retrieves the API port number from the configuration."""
    port = APP_CONFIG.get("api_port", 9012)
    try:
        return int(port)
    except (ValueError, TypeError):
        logger.warning(f"Invalid api_port value '{port}' in config. Using default 9012.")
        return 9012

def get_runner_strategy() -> str:
    """Retrieves the runner strategy, defaulting to 'all'."""
    strategy = APP_CONFIG.get("runner_strategy", "all")
    if strategy not in ["all", "single", "first_n"]:
        logger.warning(f"Invalid runner_strategy '{strategy}' in config. Defaulting to 'all'.")
        return "all"
    return strategy

def get_runner_task_id() -> Optional[str]:
    """Retrieves the specific task ID for the 'single' strategy."""
    return APP_CONFIG.get("runner_task_id")

def get_runner_first_n() -> Optional[int]:
    """Retrieves the number of tasks for the 'first_n' strategy."""
    n = APP_CONFIG.get("runner_first_n")
    if n is not None:
        try:
            val = int(n)
            return val if val > 0 else None
        except (ValueError, TypeError):
            logger.warning(f"Invalid runner_first_n value '{n}' in config. Ignoring.")
            return None
    return None

def get_runner_max_retries() -> int:
    """Retrieves the maximum number of retries for failed tasks."""
    retries = APP_CONFIG.get("runner_max_retries", 0)
    try:
        val = int(retries)
        return max(0, val)
    except (ValueError, TypeError):
        logger.warning(f"Invalid runner_max_retries value '{retries}' in config. Using default 0.")
        return 0

def get_runner_max_workers() -> int:
    """Retrieves the maximum number of concurrent workers for running tasks."""
    workers = APP_CONFIG.get("runner_max_workers", 1)
    try:
        val = int(workers)
        return max(1, val)
    except (ValueError, TypeError):
        logger.warning(f"Invalid runner_max_workers value '{workers}' in config. Using default 1.")
        return 1

# --- 确保在模块加载时打印日志 ---
if not APP_CONFIG:
    # 这个分支现在理论上不会执行，因为上面有 fallback
    logger.critical("Failed to load configuration. Please check config.json.")
else:
    # 打印一些关键配置以确认加载正确
    logger.info(f"Orchestrator model from config: {get_model('orchestrator_model')}")
    logger.info(f"Specialist Flash model from config: {get_model('specialist_model_flash')}") # 添加更多打印
    logger.info(f"Specialist Pro model from config: {get_model('specialist_model_pro')}")
    logger.info(f"GAIA Data Directory from config: {get_gaia_data_dir()}")
    logger.info(f"API Port from config: {get_api_port()}")
    logger.info(f"Runner Strategy: {get_runner_strategy()}")
    logger.info(f"Runner Max Retries: {get_runner_max_retries()}")
    logger.info(f"Runner Max Workers: {get_runner_max_workers()}")