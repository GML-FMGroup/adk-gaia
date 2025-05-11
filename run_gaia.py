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

# 不可靠答案列表
UNRELIABLE_ANSWERS = [None, "[Agent could not determine the answer]", "[Agent did not provide a final answer]"]
logger.info(f"UNRELIABLE_ANSWERS defined as: {UNRELIABLE_ANSWERS}")

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
                        logger.warning(f"Skipping line {i + 1} due to missing 'task_id': {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i + 1} in {metadata_file}: {line.strip()}")
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
        return summary[-5:] if len(summary) > 5 else summary
    except Exception as e:
        logger.error(f"Error formatting reasoning trace: {e}", exc_info=True)
        return ["Error formatting trace."]


# --- API 调用与重试逻辑 ---
def call_agent_api(task_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
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
            if os.path.commonpath([abs_gaia_split_dir]) == os.path.commonpath(
                    [abs_gaia_split_dir, potential_path]) and os.path.isfile(potential_path):
                question += f"\n\n[System Note: Absolute path for '{file_name}' is: {potential_path}]"
                if attempt == 0: logger.info(f"Appended absolute file path for task {task_id}: {potential_path}")
            elif attempt == 0:
                logger.warning(
                    f"File '{file_name}' invalid or outside expected directory '{abs_gaia_split_dir}' for task {task_id}.")
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
        # non_fatal_api_error_occurred = False # 不再单独使用此标志来决定重试

        start_time = time.time()

        try:
            response = requests.post(CHAT_ENDPOINT, json=request_payload, timeout=600)
            current_status_code = response.status_code
            final_result["api_response_status"] = current_status_code

            if current_status_code >= 400:
                error_text = response.text[:500]
                logger.error(
                    f"API request failed task {task_id} attempt {current_attempt_num} status {current_status_code}. Response: {error_text}")
                current_error_message = f"API Error {current_status_code}"
                # 对于新的重试逻辑，所有API错误都可能导致重试（除非达到最大次数）
                # response.raise_for_status() # 不需要主动抛出，让逻辑在后面决定是否重试

            if not current_error_message:
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

            # --- 实时评分 ---
            if api_call_succeeded_with_data and not current_error_message and EVAL_AVAILABLE and ground_truth is not None and current_model_answer is not None:
                logger.info(f"Scoring task {task_id} attempt {current_attempt_num}...")
                try:
                    is_correct_val = question_scorer(current_model_answer, ground_truth)
                    current_is_correct = is_correct_val
                    final_result["is_correct"] = is_correct_val
                except Exception as score_err:
                    logger.error(f"Error scoring task {task_id} attempt {current_attempt_num}: {score_err}", exc_info=True)
                    current_is_correct = "Scoring Error"
                    final_result["is_correct"] = current_is_correct
            elif api_call_succeeded_with_data and not current_error_message:  # API 调用成功，但可能不评分
                if not EVAL_AVAILABLE:
                    current_is_correct = "Scoring N/A (eval.py not imported)"
                elif ground_truth is None:
                    current_is_correct = "Scoring N/A (GT missing or Test Set)"
                # 如果 current_model_answer 是 None 但 API 成功且无 agent error，这里 current_is_correct 会是 None
                # 我们会在重试逻辑中处理这种情况
                final_result["is_correct"] = current_is_correct


        except requests.exceptions.Timeout:
            logger.error(f"API request timed out task {task_id} attempt {current_attempt_num}")
            current_error_message = "API request timed out"
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed task {task_id} attempt {current_attempt_num}: {e}")
            current_error_message = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                current_status_code = e.response.status_code
                final_result["api_response_status"] = current_status_code
        except json.JSONDecodeError as e:
            response_text = response.text if 'response' in locals() and hasattr(response, 'text') else "N/A"
            status_code = current_status_code or final_result.get('api_response_status') or 'N/A'
            logger.error(
                f"Failed to decode API JSON response task {task_id} attempt {current_attempt_num}. Status: {status_code}, Response: {response_text[:200]}... Error: {e}")
            current_error_message = f"Failed to decode API response (status {status_code})"
        except Exception as e:
            logger.exception(f"Unexpected error task {task_id} attempt {current_attempt_num}: {e}")
            current_error_message = f"Unexpected error: {str(e)}"
        finally:
            duration = time.time() - start_time
            status_msg = "API Call OK" if api_call_succeeded_with_data and not current_error_message else "API Call/Processing Failed"
            correct_log_msg = f"Correct: {current_is_correct}" if current_is_correct is not None else (
                "Correct: N/A" if GAIA_SPLIT == "validation" else "")
            answer_preview = str(
                current_model_answer if current_model_answer is not None else final_result.get("model_answer"))[:100]
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

            # --- 修正后的重试逻辑 ---

            # 1. 停止条件：
            #   a. 对于验证集，如果答案明确正确。
            #   b. 对于测试集，如果答案不是“不可靠答案”且当次尝试没有错误。
        if GAIA_SPLIT == "validation" and current_is_correct is True:
            logger.info(f"Task {task_id} (Validation) CORRECT on attempt {current_attempt_num}. Exiting retry loop.")
            break
        elif GAIA_SPLIT == "test" and \
                current_model_answer not in UNRELIABLE_ANSWERS and \
                not current_error_message:
            logger.info(f"Task {task_id} (Test) received a reliable answer ('{str(current_model_answer)[:100]}') "
                        f"without errors on attempt {current_attempt_num}. Exiting retry loop.")
            break

            # 2. 如果这是最后一次尝试，记录最终状态并退出循环
        if current_attempt_num == TOTAL_TRIES:
            log_message_end_attempts = f"Task {task_id} ({GAIA_SPLIT}) reached max attempts ({TOTAL_TRIES}). "
            if GAIA_SPLIT == "validation":
                log_message_end_attempts += f"Final score: {current_is_correct}. "
            log_message_end_attempts += f"Final answer: '{str(current_model_answer)[:100]}'. Error (if any): '{current_error_message}'."

            # 根据最终状态使用不同日志级别
            if GAIA_SPLIT == "validation" and current_is_correct is not True:
                logger.error(log_message_end_attempts)
            elif GAIA_SPLIT == "test" and (current_model_answer in UNRELIABLE_ANSWERS or current_error_message):
                logger.error(log_message_end_attempts)
            else:
                logger.info(log_message_end_attempts)
            break  # 退出循环，这是最后一次尝试的结果

            # 3. 如果前面没有 break，则意味着需要重试。构造重试原因。
            # (因为如果不需要重试，上面的条件已经 break 了)
        retry_reason_parts = []
        if GAIA_SPLIT == "validation":  # 验证集，答案不为 True 则重试
            if current_is_correct is False:
                retry_reason_parts.append("answer was incorrect")
            elif isinstance(current_is_correct, str):  # "Scoring Error" or "Scoring N/A..."
                retry_reason_parts.append(f"scoring status is '{current_is_correct}'")
            elif current_is_correct is None:  # 比如 model_answer 是 None 导致未评分
                retry_reason_parts.append("answer was not scorable to True (e.g. None or unreliable)")

        # 对两种模式都适用：如果答案不可靠或有错误，也是重试的原因
        if current_model_answer in UNRELIABLE_ANSWERS:
            if not any("unreliable" in p for p in retry_reason_parts):  # 避免重复添加
                retry_reason_parts.append(f"answer '{str(current_model_answer)[:50]}' is unreliable")

        if current_error_message:  # 任何 API 或 Agent 错误
            if not any("error occurred" in p for p in retry_reason_parts):  # 避免重复添加
                retry_reason_parts.append(f"error occurred: '{current_error_message}'")

        # 如果到这里 retry_reason_parts 还是空的，说明验证集 current_is_correct 不是 True 也不是明确的 False/Error
        # 或者是测试集，虽然答案可靠且无错误，但之前的 break 条件没满足（这不应该发生）
        # 为了保险起见，给一个默认原因
        if not retry_reason_parts:
            if GAIA_SPLIT == "validation" and current_is_correct is not True:
                retry_reason_parts.append(f"validation condition not met (score: {current_is_correct})")
            elif GAIA_SPLIT == "test":  # 对于测试集，如果到这里，表示逻辑可能存在未覆盖的边缘情况
                retry_reason_parts.append("test set general retry condition")

        final_retry_reason = "; ".join(filter(None, retry_reason_parts))
        if not final_retry_reason:  # 再次确保有原因
            final_retry_reason = "implicit condition for retry"

        logger.warning(f"Task {task_id} ({GAIA_SPLIT}) attempt {current_attempt_num} "
                       f"did not achieve target state (Reason: {final_retry_reason}). "
                       f"Retrying ({current_attempt_num + 1}/{TOTAL_TRIES})...")
        time.sleep(2)  # 可选的延迟
        # for 循环将自动进入下一次尝试

    return final_result


# --- run_evaluation_script 函数 ---
def run_evaluation_script(results_filepath: str, metadata_filepath: str) -> Optional[str]:
    logger.info(f"Attempting to run final evaluation using eval.py...")
    eval_script_path = os.path.join(os.path.dirname(__file__), "eval.py")
    if not os.path.exists(eval_script_path):
        logger.error(f"eval.py script not found at {eval_script_path}. Cannot run final evaluation.")
        return None

    command = [sys.executable, eval_script_path, results_filepath, "--metadata_file", metadata_filepath]
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        logger.info("--- eval.py Output ---")
        print(process.stdout)  # 打印到控制台
        if process.stderr:
            logger.warning("--- eval.py Stderr ---")
            print(process.stderr)  # 打印到控制台
        logger.info("--- End of eval.py Output ---")
        return process.stdout  # 返回标准输出用于写入文件
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
    else:  # "all" or any other value defaults to all
        tasks_to_run = all_tasks_list
        logger.info(f"Selected all {len(tasks_to_run)} tasks.")

    if not tasks_to_run: logger.warning("No tasks selected to run."); return

    try:
        logger.info(f"Pinging API server at {API_BASE_URL}...")
        ping_url = f"{API_BASE_URL}/docs"  # Assuming FastAPI /docs endpoint for ping
        requests.get(ping_url, timeout=10).raise_for_status()
        logger.info(f"Successfully connected to API at {API_BASE_URL}")
    except requests.exceptions.RequestException as e:
        logger.critical(f"Error connecting to API at {API_BASE_URL} ({ping_url}): {e}. Is the server running?")
        return

    output_file_path = OUTPUT_FILE_TEMPLATE.format(timestamp=int(time.time()))
    processed_count = 0
    logger.info(
        f"Starting processing {len(tasks_to_run)} tasks with {MAX_WORKERS} worker(s). Results -> {output_file_path}")

    evaluation_summary_output = None

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(call_agent_api, task_data, f"{USER_ID_PREFIX}_{task_data['task_id']}"): task_data['task_id']
                    for task_data in tasks_to_run
                }

                for future in as_completed(futures):
                    task_id_completed = futures[future]
                    try:
                        result = future.result()
                        original_task_data = gaia_metadata_dict.get(task_id_completed, {})
                        result["ground_truth"] = original_task_data.get("ground_truth")
                        result["level"] = original_task_data.get("Level")

                        if not isinstance(result.get("is_correct"), (bool, str, type(None))):
                            logger.warning(
                                f"Converting non-standard is_correct for task {task_id_completed} to string: {result.get('is_correct')}")
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

            if processed_count > 0 and os.path.exists(output_file_path) and EVAL_AVAILABLE and GAIA_SPLIT == "validation":
                logger.info(f"--- Running Final Evaluation on {output_file_path} (Validation Set) ---")
                evaluation_summary_output = run_evaluation_script(output_file_path, metadata_file)

                if evaluation_summary_output:
                    logger.info(f"Appending evaluation summary to {output_file_path}")
                    summary_entry = {
                        "entry_type": "evaluation_summary",
                        "timestamp": int(time.time()),
                        "source_file": os.path.basename(output_file_path),
                        "summary_text": evaluation_summary_output.strip().split('\n')
                    }
                    with open(output_file_path, 'a', encoding='utf-8') as summary_outfile:
                        summary_outfile.write(json.dumps(summary_entry) + '\n')
                        summary_outfile.flush()
            elif GAIA_SPLIT == "test":
                logger.info("Skipping final evaluation summary for test set.")
            elif not EVAL_AVAILABLE:
                logger.warning("EVAL_AVAILABLE is False, skipping final evaluation summary.")
            else:
                logger.warning("No tasks were processed or results file not found, skipping final evaluation summary.")

        except Exception as e:
            logger.critical(f"An unexpected error occurred during task execution phase: {e}", exc_info=True)

    logger.info("--- GAIA Agent Runner Finished ---")


if __name__ == "__main__":
    if not GAIA_SPLIT_DIR:
        logger.critical(
            "GAIA data directory path could not be determined. Check config.json or ensure GAIA_DATA_DIR env var is set. Exiting.")
    else:
        main()