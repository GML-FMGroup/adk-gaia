# run_gaia.py
import json
import os
import sys
import time
import requests
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import subprocess

# --- 从配置模块导入 ---
try:
    from src.core.config import (
        get_gaia_data_dir,
        get_api_port,
        get_runner_strategy,
        get_runner_task_id,
        get_runner_first_n,
        get_runner_max_retries,
        get_runner_max_workers,
        get_gaia_split
    )
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Error importing configuration: {e}. Make sure src/core/config.py exists and is correct.")
    get_gaia_data_dir = lambda: "./GAIA/2023"
    get_api_port = lambda: 9012
    get_runner_strategy = lambda: "all"
    get_runner_task_id = lambda: None
    get_runner_first_n = lambda: None
    get_runner_max_retries = lambda: 0
    get_runner_max_workers = lambda: 1
    get_gaia_split = lambda: "validation"
    logger = logging.getLogger(__name__)
    logger.warning("Using default configuration values due to import error.")

# --- 从 eval.py 导入评分函数 ---
try:
    from eval import question_scorer
    EVAL_AVAILABLE = True
    logger.info("Successfully imported scoring functions from eval.py for real-time evaluation.")
except ImportError:
    logger.error("Could not import scoring functions from eval.py. Real-time evaluation disabled.")
    EVAL_AVAILABLE = False
    def question_scorer(model_answer: Optional[str], ground_truth: Optional[str]) -> Optional[bool]:
        logger.warning("question_scorer stub called because eval import failed.")
        return None

# --- 配置日志 ---
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- 常量和配置加载 ---
API_HOST = os.getenv("API_HOST", "http://localhost")
API_PORT = get_api_port()
API_BASE_URL = f"{API_HOST}:{API_PORT}"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

GAIA_SPLIT = get_gaia_split()
logger.info(f"GAIA_SPLIT set to: {GAIA_SPLIT}")


OUTPUT_FILE_TEMPLATE = f"gaia_{GAIA_SPLIT}_results_{{timestamp}}.jsonl"
USER_ID_PREFIX = f"gaia_runner_{GAIA_SPLIT}"

RUNNER_STRATEGY = get_runner_strategy()
RUNNER_TASK_ID = get_runner_task_id()
RUNNER_FIRST_N = get_runner_first_n()
MAX_RETRIES = get_runner_max_retries()
MAX_WORKERS = get_runner_max_workers()

TOTAL_TRIES = MAX_RETRIES + 1

GAIA_BASE_DIR = get_gaia_data_dir()
GAIA_SPLIT_DIR = os.path.join(GAIA_BASE_DIR, GAIA_SPLIT) if GAIA_BASE_DIR else None

# --- 加载 GAIA 数据 ---
def load_gaia_metadata(metadata_file: str) -> Dict[str, Dict[str, Any]]:
    if not GAIA_SPLIT_DIR or not os.path.exists(metadata_file):
        logger.error(f"GAIA metadata file not found at {metadata_file} or directory not configured.")
        return {}
    tasks_dict = {}
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    task_data = json.loads(line)
                    task_id = task_data.get("task_id")
                    if task_id:
                        if GAIA_SPLIT == "validation":
                            task_data["ground_truth"] = task_data.get("Final Answer") or task_data.get("Final answer")
                        else:
                            task_data["ground_truth"] = None
                        tasks_dict[task_id] = task_data
                    else:
                         logger.warning(f"Skipping line {i+1} due to missing 'task_id': {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in {metadata_file}: {line.strip()}")
        logger.info(f"Loaded {len(tasks_dict)} tasks from {metadata_file} for split '{GAIA_SPLIT}'")
        return tasks_dict
    except Exception as e:
        logger.exception(f"Error reading GAIA metadata file {metadata_file}: {e}")
        return {}

def format_reasoning_trace(events: Optional[List[Dict[str, Any]]]) -> Optional[List[str]]:
    if not events:
        return None
    summary = []
    try:
        for event in events:
            author = event.get("author", "UnknownAgent")
            content = event.get("content", {})
            parts = content.get("parts", [])
            event_summary = f"Agent: {author}"

            if event.get("is_final_response") and parts and "text" in parts[0] and parts[0].get("text"):
                 event_summary += f" -> FINAL RESPONSE: {str(parts[0]['text'])[:150]}..."
            elif parts:
                part = parts[0]
                if "text" in part and part.get("text"):
                    event_summary += f" -> Text: {str(part['text'])[:100]}..."
                elif "function_call" in part:
                    fc = part["function_call"]
                    args_str = str(fc.get('args', {}))
                    event_summary += f" -> Calls Function: {fc.get('name')}(args={args_str[:100]}{'...' if len(args_str) > 100 else ''})"
                elif "function_response" in part:
                    fr = part["function_response"]
                    response_str = str(fr.get('response', {}))
                    event_summary += f" -> Function Result: {fr.get('name')} -> {response_str[:100]}{'...' if len(response_str) > 100 else ''}"
                else:
                    event_summary += " -> Non-text part or unknown content structure"
            else:
                event_summary += " -> No content parts"
            summary.append(event_summary)
        return summary[-5:]
    except Exception as e:
        logger.error(f"Error formatting reasoning trace: {e}", exc_info=True)
        return ["Error formatting trace."]

# --- API 调用与重试逻辑 ---
def call_agent_api(task_data: Dict[str, Any], user_id: str) -> Dict[str, Any]: # user_id is now a required argument
    task_id = task_data["task_id"]
    original_question = task_data["Question"]
    file_name = task_data.get("file_name")
    ground_truth = task_data.get("ground_truth")
    task_level = task_data.get("Level")

    final_result = {
        "task_id": task_id,
        "level": task_level,
        "model_answer": None,
        "ground_truth": ground_truth,
        "is_correct": None,
        "attempts": 0,
        "reasoning_trace": None,
        "error": None,
        "api_response_status": None,
    }

    for attempt in range(TOTAL_TRIES):
        current_attempt_num = attempt + 1
        final_result["attempts"] = current_attempt_num
        session_id = f"session_{task_id}_attempt_{current_attempt_num}_{uuid.uuid4()}"
        logger.info(f"--- Task {task_id} (Level {task_level}): Attempt {current_attempt_num}/{TOTAL_TRIES} ---")

        question = original_question
        if file_name and GAIA_SPLIT_DIR:
            abs_gaia_split_dir = os.path.abspath(GAIA_SPLIT_DIR)
            potential_path = os.path.abspath(os.path.join(abs_gaia_split_dir, file_name))
            if os.path.commonpath([abs_gaia_split_dir]) == os.path.commonpath([abs_gaia_split_dir, potential_path]) and os.path.isfile(potential_path):
                question += f"\n\n[System Note: Absolute path for '{file_name}' is: {potential_path}]"
                if attempt == 0: logger.info(f"Appended absolute file path for task {task_id}: {potential_path}")
            elif attempt == 0:
                logger.warning(f"File '{file_name}' invalid or outside expected directory '{abs_gaia_split_dir}' for task {task_id}.")
        elif file_name and attempt == 0:
             logger.warning(f"File name '{file_name}' provided for task {task_id}, but GAIA_SPLIT_DIR not valid.")

        request_payload = {
            "user_id": user_id, "session_id": session_id, "task_id": task_id,
            "question": question, "file_name": file_name
        }
        logger.debug(f"Sending payload attempt {current_attempt_num}: {request_payload}")

        current_error_message = None
        current_model_answer = None
        full_reasoning_events = None
        current_status_code = None
        current_is_correct = None
        api_call_succeeded_with_data = False
        non_fatal_api_error_occurred = False

        start_time = time.time()

        try:
            response = requests.post(CHAT_ENDPOINT, json=request_payload, timeout=600)
            current_status_code = response.status_code
            final_result["api_response_status"] = current_status_code

            if current_status_code >= 400:
                 error_text = response.text[:500]
                 logger.error(f"API request failed task {task_id} attempt {current_attempt_num} status {current_status_code}. Response: {error_text}")
                 current_error_message = f"API Error {current_status_code}"
                 if 500 <= current_status_code <= 599 or current_status_code in [408, 429]:
                     non_fatal_api_error_occurred = True
                 else: # Fatal 4xx error
                     response.raise_for_status() # Will be caught by RequestException

            if not current_error_message : # Only proceed if no HTTP error yet
                response_data = response.json()
                api_call_succeeded_with_data = True
                current_model_answer = response_data.get("model_answer")
                current_agent_error = response_data.get("error")
                full_reasoning_events = response_data.get("reasoning_trace")

                final_result["model_answer"] = current_model_answer
                final_result["reasoning_trace"] = format_reasoning_trace(full_reasoning_events)

                if current_agent_error:
                     logger.error(f"Agent returned error for task {task_id} attempt {current_attempt_num}: {current_agent_error}")
                     current_error_message = f"Agent Error: {current_agent_error}"
                     # Assume agent errors are fatal for retry purposes for now
                     non_fatal_api_error_occurred = False # Override to false if agent error

            # --- 实时评分 ---
            if api_call_succeeded_with_data and not current_error_message and EVAL_AVAILABLE and ground_truth is not None and current_model_answer is not None:
                 logger.info(f"Scoring task {task_id} attempt {current_attempt_num}...")
                 try:
                    is_correct_val = question_scorer(current_model_answer, ground_truth)
                    current_is_correct = is_correct_val
                    final_result["is_correct"] = is_correct_val
                    if is_correct_val: break
                 except Exception as score_err:
                      logger.error(f"Error scoring task {task_id} attempt {current_attempt_num}: {score_err}", exc_info=True)
                      current_is_correct = "Scoring Error"
                      final_result["is_correct"] = current_is_correct
            elif api_call_succeeded_with_data and not current_error_message :
                 if not EVAL_AVAILABLE: current_is_correct = "Scoring N/A (eval.py not imported)"
                 elif ground_truth is None: current_is_correct = "Scoring N/A (GT missing or Test Set)"
                 elif current_model_answer is None: current_is_correct = False
                 final_result["is_correct"] = current_is_correct

        except requests.exceptions.Timeout:
            logger.error(f"API request timed out task {task_id} attempt {current_attempt_num}")
            current_error_message = "API request timed out"; non_fatal_api_error_occurred = True
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed task {task_id} attempt {current_attempt_num}: {e}")
            current_error_message = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                current_status_code = e.response.status_code
                final_result["api_response_status"] = current_status_code
                if 500 <= current_status_code <= 599 or current_status_code in [408, 429]:
                    non_fatal_api_error_occurred = True
            else: non_fatal_api_error_occurred = True
        except json.JSONDecodeError as e:
            response_text = response.text if 'response' in locals() and hasattr(response, 'text') else "N/A"
            status_code = current_status_code or final_result.get('api_response_status') or 'N/A'
            logger.error(f"Failed to decode API JSON response task {task_id} attempt {current_attempt_num}. Status: {status_code}, Response: {response_text[:200]}... Error: {e}")
            current_error_message = f"Failed to decode API response (status {status_code})"
            non_fatal_api_error_occurred = True
        except Exception as e:
            logger.exception(f"Unexpected error task {task_id} attempt {current_attempt_num}: {e}")
            current_error_message = f"Unexpected error: {str(e)}"; non_fatal_api_error_occurred = True
        finally:
            duration = time.time() - start_time
            status_msg = "API Call OK" if api_call_succeeded_with_data and not current_error_message else "API Call/Processing Failed"
            correct_log_msg = f"Correct: {current_is_correct}" if current_is_correct is not None else ("Correct: N/A" if GAIA_SPLIT == "validation" else "")
            answer_preview = str(current_model_answer if current_model_answer is not None else final_result.get("model_answer"))[:100]
            if current_model_answer is not None and len(str(current_model_answer)) > 100: answer_preview += '...'
            log_message = (
                f"Task {task_id} attempt {current_attempt_num} finished in {duration:.2f}s. "
                f"Status: {status_msg}. {correct_log_msg} "
                f"Model Answer: '{answer_preview}'. "
            )
            if GAIA_SPLIT == "validation": log_message += f"Ground Truth: '{ground_truth}'. "
            log_message += f"Error this attempt: {current_error_message}"
            logger.info(log_message)

        if current_error_message:
            final_result["error"] = current_error_message
        if final_result["is_correct"] is None and not (api_call_succeeded_with_data and not current_error_message):
             final_result["is_correct"] = False

        # --- 重试逻辑 ---
        should_retry = False
        if current_is_correct is True: # 成功，不重试
            logger.info(f"Task {task_id} CORRECT on attempt {current_attempt_num}. Exiting retry loop.")
            break
        elif attempt < MAX_RETRIES: # 如果还有重试机会
            if GAIA_SPLIT == "validation":
                # 对于验证集：如果答案不正确或评分出错，并且错误是可重试的
                if current_is_correct is False or isinstance(current_is_correct, str): # "Scoring Error" or "N/A"
                    if non_fatal_api_error_occurred or not current_error_message : # 可重试的API错误，或者没有错误但答案不对
                        should_retry = True
                        retry_reason = "retriable API error" if non_fatal_api_error_occurred else "incorrect answer/scoring issue"
            elif GAIA_SPLIT == "test":
                # 对于测试集：如果模型答案为空或API调用可重试
                if current_model_answer is None and non_fatal_api_error_occurred:
                    should_retry = True
                    retry_reason = "retriable API error (no answer)"
                elif current_model_answer is None and not current_error_message: # 没答案但也没致命错误
                    should_retry = True
                    retry_reason = "no model answer"
            if should_retry:
                logger.warning(f"Task {task_id} attempt {current_attempt_num} resulted in {retry_reason}. Retrying ({current_attempt_num + 1}/{TOTAL_TRIES})...")
                time.sleep(2) # Optional delay
                continue # 继续下一次尝试
            else: # 不可重试的错误，或者测试集已有答案
                logger.info(f"Exiting retry loop for task {task_id} after attempt {current_attempt_num} due to non-retriable condition or test set answer received.")
                break
        else: # 已达到最大尝试次数
             logger.error(f"Task {task_id} still not resolved after {TOTAL_TRIES} attempts.")
             break # 确保退出循环

    return final_result

# --- run_evaluation_script 函数 (保持不变) ---
def run_evaluation_script(results_filepath: str, metadata_filepath: str) -> Optional[str]:
    # ... (代码同上一个版本) ...
    logger.info(f"Attempting to run final evaluation using eval.py...")
    eval_script_path = os.path.join(os.path.dirname(__file__), "eval.py")
    if not os.path.exists(eval_script_path):
        logger.error(f"eval.py script not found at {eval_script_path}. Cannot run final evaluation.")
        return None

    command = [ sys.executable, eval_script_path, results_filepath, "--metadata_file", metadata_filepath]
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        logger.info("--- eval.py Output ---")
        print(process.stdout)
        if process.stderr:
            logger.warning("--- eval.py Stderr ---")
            print(process.stderr)
        logger.info("--- End of eval.py Output ---")
        return process.stdout
    except FileNotFoundError:
         logger.error(f"Error: Python executable '{sys.executable}' or eval script '{eval_script_path}' not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"eval.py script failed with return code {e.returncode}:")
        logger.error(f"--- eval.py Stdout ---\n{e.stdout}")
        logger.error(f"--- eval.py Stderr ---\n{e.stderr}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while running eval.py: {e}", exc_info=True)
    return None

# --- main 函数 ---
def main():
    logger.info(f"--- Starting GAIA Agent Runner ({GAIA_SPLIT.upper()}) ---")
    logger.info(f"Max Retries: {MAX_RETRIES}, Total Tries per Task: {TOTAL_TRIES}")

    if not GAIA_SPLIT_DIR: logger.critical("GAIA data directory path not determined. Exiting."); return
    metadata_file = os.path.join(GAIA_SPLIT_DIR, "metadata.jsonl")
    gaia_metadata_dict = load_gaia_metadata(metadata_file)
    if not gaia_metadata_dict: logger.critical("No tasks loaded from metadata. Exiting."); return

    all_tasks_list = list(gaia_metadata_dict.values())
    tasks_to_run: List[Dict[str, Any]] = []
    logger.info(f"Applying runner strategy: {RUNNER_STRATEGY}")

    if RUNNER_STRATEGY == "single":
        if not RUNNER_TASK_ID: logger.critical("Strategy 'single' requires 'runner_task_id' in config. Exiting."); return
        single_task_data = gaia_metadata_dict.get(RUNNER_TASK_ID)
        if not single_task_data: logger.error(f"Task ID '{RUNNER_TASK_ID}' not found. Exiting."); return
        tasks_to_run = [single_task_data]
        logger.info(f"Selected single task: {RUNNER_TASK_ID}")
    elif RUNNER_STRATEGY == "first_n":
        n = get_runner_first_n()
        if n is None or n <= 0: logger.critical("Strategy 'first_n' requires positive 'runner_first_n'. Exiting."); return
        tasks_to_run = all_tasks_list[:n]
        logger.info(f"Selected first {len(tasks_to_run)} tasks (requested: {n}).")
    else:
        tasks_to_run = all_tasks_list
        logger.info(f"Selected all {len(tasks_to_run)} tasks.")

    if not tasks_to_run: logger.warning("No tasks selected to run."); return

    try:
        logger.info(f"Pinging API server at {API_BASE_URL}...")
        ping_url = f"{API_BASE_URL}/docs"
        requests.get(ping_url, timeout=10).raise_for_status()
        logger.info(f"Successfully connected to API at {API_BASE_URL}")
    except requests.exceptions.RequestException as e:
        logger.critical(f"Error connecting to API at {API_BASE_URL} ({ping_url}): {e}. Is the server running?")
        return

    output_file_path = OUTPUT_FILE_TEMPLATE.format(timestamp=int(time.time()))
    processed_count = 0
    logger.info(f"Starting processing {len(tasks_to_run)} tasks with {MAX_WORKERS} worker(s). Results -> {output_file_path}")

    evaluation_summary_output = None

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # **修正：确保传递 user_id**
                futures = {
                    executor.submit(call_agent_api, task_data, f"{USER_ID_PREFIX}_{task_data['task_id']}"): task_data['task_id']
                    for task_data in tasks_to_run
                }

                for future in as_completed(futures):
                    task_id_completed = futures[future]
                    try:
                        result = future.result()
                        # 从 gaia_metadata_dict 获取原始 task_data 以确保 GT 和 Level 最新
                        # (因为 result 字典是 call_agent_api 内部构建的)
                        original_task_data = gaia_metadata_dict.get(task_id_completed, {})
                        result["ground_truth"] = original_task_data.get("ground_truth")
                        result["level"] = original_task_data.get("Level")


                        if not isinstance(result.get("is_correct"), (bool, str, type(None))):
                            logger.warning(f"Converting non-standard is_correct for task {task_id_completed} to string: {result.get('is_correct')}")
                            result["is_correct"] = str(result.get("is_correct"))

                        outfile.write(json.dumps(result, default=str) + '\n')
                        outfile.flush()
                        processed_count += 1
                    except Exception as e:
                        logger.exception(f"Critical error retrieving/writing result for task {task_id_completed}: {e}")
                        task_info_for_error = gaia_metadata_dict.get(task_id_completed, {})
                        error_result = {
                            "task_id": task_id_completed, "model_answer": None,
                            "ground_truth": task_info_for_error.get("ground_truth"),
                            "level": task_info_for_error.get("Level"),
                            "is_correct": False, "attempts": TOTAL_TRIES,
                            "error": f"Future processing/writing failed: {str(e)}",
                            "api_response_status": None,
                        }
                        try:
                            outfile.write(json.dumps(error_result) + '\n')
                            outfile.flush()
                        except Exception as write_err:
                             logger.error(f"Failed to write error result for task {task_id_completed} to file: {write_err}")
                        processed_count += 1
                logger.info(f"All submitted tasks futures completed (processed {processed_count} tasks).")

            # --- Final Evaluation and Appending Summary ---
            if processed_count > 0 and os.path.exists(output_file_path):
                logger.info(f"--- Running Final Evaluation on {output_file_path} ---")
                # 注意：这里在 with outfile 块外部调用，所以 run_evaluation_script 不能直接写入 outfile
                # run_evaluation_script 应该返回字符串，然后我们再追加
                evaluation_summary_output = run_evaluation_script(output_file_path, metadata_file)

                if evaluation_summary_output:
                    logger.info(f"Appending evaluation summary to {output_file_path}")
                    summary_entry = {
                        "entry_type": "evaluation_summary",
                        "timestamp": int(time.time()),
                        "source_file": os.path.basename(output_file_path),
                        "summary_text": evaluation_summary_output.strip().split('\n')
                    }
                    # 重新以追加模式打开文件
                    with open(output_file_path, 'a', encoding='utf-8') as summary_outfile:
                        summary_outfile.write(json.dumps(summary_entry) + '\n')
                        summary_outfile.flush()
            else:
                logger.warning("No tasks were processed or results file not found, skipping final evaluation.")

        except Exception as e:
            logger.critical(f"An unexpected error occurred during task execution phase: {e}", exc_info=True)

    logger.info("--- GAIA Agent Runner Finished ---")


if __name__ == "__main__":
    if not GAIA_SPLIT_DIR:
        logger.critical("GAIA data directory path could not be determined. Check config.json. Exiting.")
    else:
        main()