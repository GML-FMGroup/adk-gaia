# src/core/debug_logger.py
import logging
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import threading

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai.types import Content, Part

from .config import should_save_debug_chat_log, get_gaia_data_dir

logger = logging.getLogger(__name__)

GAIA_BASE_DIR = get_gaia_data_dir()
DEBUG_LOG_DIR_NAME = "debug_chat_logs_json_by_task"

if GAIA_BASE_DIR:
  DEBUG_LOG_DIR = os.path.join(GAIA_BASE_DIR, DEBUG_LOG_DIR_NAME)
  if not os.path.exists(DEBUG_LOG_DIR):
    try:
      os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
      logger.info(f"Created debug chat log (JSON by task) directory: {DEBUG_LOG_DIR}")
    except OSError as e:
      logger.error(f"Failed to create debug chat log (JSON by task) directory {DEBUG_LOG_DIR}: {e}")
      DEBUG_LOG_DIR = None
  else:
    logger.info(f"Debug chat log (JSON by task) directory already exists: {DEBUG_LOG_DIR}")
else:
  logger.warning("GAIA_DATA_DIR not configured. Debug chat logs (JSON by task) will not be saved.")
  DEBUG_LOG_DIR = None

_file_locks: Dict[str, threading.Lock] = {}
_lock_for_locks_dict = threading.Lock()


def _get_file_lock(filepath: str) -> threading.Lock:
  with _lock_for_locks_dict:
    if filepath not in _file_locks:
      _file_locks[filepath] = threading.Lock()
    return _file_locks[filepath]


def _safe_model_dump(obj: Any) -> Any:
  if hasattr(obj, 'model_dump_json'):
    try:
      return json.loads(obj.model_dump_json(exclude_none=True, warnings=False))
    except Exception:
      pass
  if hasattr(obj, 'model_dump'):
    try:
      return obj.model_dump(exclude_none=True, warnings=False)
    except Exception as e_dump:
      logger.debug(f"Could not model_dump object of type {type(obj)}: {e_dump}")
      return str(obj)
  return obj


def _ensure_serializable(data: Any) -> Any:
  if isinstance(data, (str, int, float, bool, type(None))):
    return data
  elif isinstance(data, dict):
    return {str(k): _ensure_serializable(v) for k, v in data.items()}
  elif isinstance(data, list):
    return [_ensure_serializable(i) for i in data]
  elif isinstance(data, (Content, Part)):
    try:
      return _ensure_serializable(data.to_dict())
    except Exception as e:
      logger.debug(f"Could not call to_dict() on {type(data)}: {e}. Falling back to str().")
      return str(data)
  else:
    dumped = _safe_model_dump(data)
    if isinstance(dumped, (dict, list)):
      return _ensure_serializable(dumped)
    return str(dumped)


def _format_log_entry(
    agent_name: str,
    event_type: str,
    data: Any,
    session_id_from_state: Optional[str] = None,  # 从 state 中获取的 session_id
    task_id: Optional[str] = None,
    invocation_id: Optional[str] = None,
    turn_id: Optional[int] = None
) -> Dict[str, Any]:
  return {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "agent_name": agent_name,
    "session_id_from_state": session_id_from_state,  # 明确来源
    "task_id": task_id,
    "invocation_id": invocation_id,
    "turn_id": turn_id,
    "event_type": event_type,
    "data": _ensure_serializable(data)
  }


def _write_log(log_entry: Dict[str, Any]):
  if not DEBUG_LOG_DIR:
    return

  task_id = log_entry.get("task_id", "unknown_task")
  # 日志文件名现在主要基于 task_id
  log_filename = f"chat_log_task_{task_id}.json"
  log_filepath = os.path.join(DEBUG_LOG_DIR, log_filename)

  file_lock = _get_file_lock(log_filepath)
  with file_lock:
    try:
      log_data = []
      if os.path.exists(log_filepath):
        with open(log_filepath, 'r', encoding='utf-8') as f_read:
          try:
            content = f_read.read()
            if content.strip():
              log_data = json.loads(content)
            if not isinstance(log_data, list):
              logger.warning(f"Log file {log_filepath} was not a JSON list. Resetting.")
              log_data = []
          except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {log_filepath}. Backing up and starting new log.")
            backup_path = f"{log_filepath}.{datetime.now().strftime('%Y%m%d%H%M%S%f')}.bak"
            try:
              os.rename(log_filepath, backup_path)
              logger.info(f"Backed up corrupted log to {backup_path}")
            except OSError as e_rename:
              logger.error(f"Could not backup corrupted log {log_filepath}: {e_rename}")
            log_data = []

      log_data.append(log_entry)

      with open(log_filepath, 'w', encoding='utf-8') as f_write:
        json.dump(log_data, f_write, indent=2, ensure_ascii=False)

    except Exception as e:
      logger.error(f"Failed to write debug chat log to {log_filepath}: {e}", exc_info=True)


# --- ADK Callbacks ---
def adk_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
  logger.info(f"--- DEBUG: adk_before_model_callback TRIGGERED for agent: {callback_context.agent_name} ---")
  if not should_save_debug_chat_log():
    logger.info("Debug log saving is disabled by config for adk_before_model_callback.")
    return None

  try:
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id  # 这个是 LlmAgent 内部的，与 ADK Event 的 invocation_id 可能不同

    # 从 session.state 获取 task_id 和 session_id (如果 api.py 中设置了)
    task_id = callback_context.state.get("current_task_id", "unknown_task_in_state")
    session_id_from_state = callback_context.state.get("current_session_id_for_debug_log", "unknown_session_in_state")

    logger.info(
      f"--- DEBUG: Task ID '{task_id}', Session ID (from state) '{session_id_from_state}' in adk_before_model_callback for agent {agent_name} ---")

    # 使用基于 task_id 和 session_id_from_state (如果存在) 的唯一键来存储 turn_id
    # 这样即使 task_id 相同但 session_id 不同（例如重试），turn_id 也能正确计数
    turn_id_state_key = f"debug_log_turn_id_for_{task_id}_{session_id_from_state}"
    turn_id = callback_context.state.get(turn_id_state_key, 0)
    current_turn_id = turn_id + 1
    callback_context.state[turn_id_state_key] = current_turn_id

    request_data_dict = llm_request.model_dump(exclude_none=True, warnings=False)

    log_entry = _format_log_entry(
      agent_name=agent_name,
      event_type="llm_request",
      data=request_data_dict,
      session_id_from_state=session_id_from_state,
      task_id=task_id,
      invocation_id=invocation_id,
      turn_id=current_turn_id
    )
    _write_log(log_entry)
    logger.info(
      f"Logged LLM request for agent {agent_name}, task {task_id}, session_from_state {session_id_from_state}, invocation {invocation_id}, turn {current_turn_id}")
  except Exception as e:
    logger.error(f"Error in before_model_callback for agent {callback_context.agent_name}: {e}", exc_info=True)
  return None


def adk_after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> None:
  logger.info(f"--- DEBUG: adk_after_model_callback TRIGGERED for agent: {callback_context.agent_name} ---")
  if not should_save_debug_chat_log():
    logger.info("Debug log saving is disabled by config for adk_after_model_callback.")
    return

  try:
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id

    task_id = callback_context.state.get("current_task_id", "unknown_task_in_state")
    session_id_from_state = callback_context.state.get("current_session_id_for_debug_log", "unknown_session_in_state")
    logger.info(
      f"--- DEBUG: Task ID '{task_id}', Session ID (from state) '{session_id_from_state}' in adk_after_model_callback for agent {agent_name} ---")

    turn_id_state_key = f"debug_log_turn_id_for_{task_id}_{session_id_from_state}"
    turn_id = callback_context.state.get(turn_id_state_key, 0)  # 获取的是当前请求的轮次ID

    response_data_dict = llm_response.model_dump(exclude_none=True, warnings=False)

    log_entry = _format_log_entry(
      agent_name=agent_name,
      event_type="llm_response",
      data=response_data_dict,
      session_id_from_state=session_id_from_state,
      task_id=task_id,
      invocation_id=invocation_id,
      turn_id=turn_id
    )
    _write_log(log_entry)
    logger.info(
      f"Logged LLM response for agent {agent_name}, task {task_id}, session_from_state {session_id_from_state}, invocation {invocation_id}, turn {turn_id}")
  except Exception as e:
    logger.error(f"Error in after_model_callback for agent {callback_context.agent_name}: {e}", exc_info=True)