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
        get_runner_max_workers
    )
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Error importing configuration: {e}. Make sure src/core/config.py exists and is correct.")
    # Provide dummy getters if import fails
    get_gaia_data_dir = lambda: "./GAIA/2023"
    get_api_port = lambda: 9012
    get_runner_strategy = lambda: "all"
    get_runner_task_id = lambda: None
    get_runner_first_n = lambda: None
    get_runner_max_retries = lambda: 0
    get_runner_max_workers = lambda: 1
    logger = logging.getLogger(__name__)
    logger.warning("Using default configuration values due to import error.")

# --- 从 eval.py 导入评分函数 ---
try:
    # 确保 eval.py 在 Python 路径中
    from eval import question_scorer, normalize_str, normalize_number_str, split_string
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
GAIA_SPLIT = "validation"
OUTPUT_FILE_TEMPLATE = f"gaia_{GAIA_SPLIT}_results_{{timestamp}}.jsonl"
USER_ID_PREFIX = "gaia_runner_val"

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
    """Loads GAIA metadata into a dictionary keyed by task_id."""
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
                        task_data["ground_truth"] = task_data.get("Final Answer") or task_data.get("Final answer")
                        tasks_dict[task_id] = task_data
                    else:
                         logger.warning(f"Skipping line {i+1} due to missing 'task_id': {line.strip()}")
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in {metadata_file}: {line.strip()}")
        logger.info(f"Loaded {len(tasks_dict)} tasks from {metadata_file}")
        return tasks_dict
    except Exception as e:
        logger.exception(f"Error reading GAIA metadata file {metadata_file}: {e}")
        return {}

# --- API 调用与重试逻辑 ---
def call_agent_api(task: Dict[str, Any], user_id: str, ground_truth: Optional[str]) -> Dict[str, Any]:
    """Calls the FastAPI endpoint, includes ground truth, real-time scoring, and retry logic."""
    task_id = task["task_id"]
    original_question = task["Question"]
    file_name = task.get("file_name")

    final_result = {
        "task_id": task_id,
        "model_answer": None, # Stores the answer from the *last* successful or final failed attempt
        "ground_truth": ground_truth,
        "is_correct": None,
        "attempts": 0,
        "reasoning_trace_summary": None, # Stores trace from the *last* attempt
        "error": None, # Stores the error from the *last* failed attempt or the first fatal API error
        "api_response_status": None, # Stores status from the *last* attempt
    }

    for attempt in range(TOTAL_TRIES):
        current_attempt_num = attempt + 1
        final_result["attempts"] = current_attempt_num
        session_id = f"session_{task_id}_attempt_{current_attempt_num}_{uuid.uuid4()}"
        logger.info(f"--- Task {task_id}: Attempt {current_attempt_num}/{TOTAL_TRIES} ---")

        question = original_question
        gaia_file_path = None
        if file_name and GAIA_SPLIT_DIR:
            abs_gaia_split_dir = os.path.abspath(GAIA_SPLIT_DIR)
            potential_path = os.path.abspath(os.path.join(abs_gaia_split_dir, file_name))
            if os.path.commonpath([abs_gaia_split_dir]) == os.path.commonpath([abs_gaia_split_dir, potential_path]) and os.path.isfile(potential_path):
                gaia_file_path = potential_path
                question += f"\n\n[System Note: Absolute path for '{file_name}' is: {gaia_file_path}]"
                if attempt == 0: logger.info(f"Appended absolute file path for task {task_id}: {gaia_file_path}")
            elif attempt == 0:
                logger.warning(f"File '{file_name}' invalid or outside expected directory '{abs_gaia_split_dir}' for task {task_id}.")
        elif file_name and attempt == 0:
             logger.warning(f"File name '{file_name}' provided for task {task_id}, but GAIA data directory not configured correctly.")


        request_payload = {
            "user_id": user_id, "session_id": session_id, "task_id": task_id,
            "question": question, "file_name": file_name
        }
        logger.debug(f"Sending payload attempt {current_attempt_num}: {request_payload}")

        # Reset specific fields for *this* attempt's results
        current_error = None
        current_model_answer = None
        current_trace_summary = None
        current_status_code = None
        current_is_correct = None
        api_error_occurred = False

        start_time = time.time()

        try:
            response = requests.post(CHAT_ENDPOINT, json=request_payload, timeout=600)
            current_status_code = response.status_code
            final_result["api_response_status"] = current_status_code # Update last status code

            if response.status_code >= 400:
                 log_payload = request_payload.copy()
                 error_text = response.text[:500]
                 logger.error(f"API request failed task {task_id} attempt {current_attempt_num} status {response.status_code}. Payload (partial): { {k: (str(v)[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) for k, v in log_payload.items()} }. Response: {error_text}")
                 current_error = f"API Error {response.status_code}"
                 api_error_occurred = True
                 response.raise_for_status()

            response_data = response.json()
            # **关键修正: 从API响应中获取 'model_answer' 字段**
            current_model_answer = response_data.get("model_answer")
            current_agent_error = response_data.get("error")

            # 记录这次尝试的答案和追踪（无论对错，只要API成功）
            final_result["model_answer"] = current_model_answer # 覆盖，记录最后一次尝试的答案
            trace = response_data.get("reasoning_trace") # API 返回 'reasoning_trace'
            if trace and isinstance(trace, list) and trace:
                 try:
                     current_trace_summary = [f"{evt.get('author', 'Unk')}:{evt.get('type', type(evt).__name__)}" for evt in trace[-3:]] # 使用 type 获取事件类型名
                     final_result["reasoning_trace_summary"] = current_trace_summary
                 except Exception: final_result["reasoning_trace_summary"] = "Error summarizing trace"

            if current_agent_error:
                 logger.error(f"Agent returned error for task {task_id} attempt {current_attempt_num}: {current_agent_error}")
                 current_error = f"Agent Error: {current_agent_error}"
                 api_error_occurred = True

            # --- 实时评分 ---
            if not api_error_occurred and EVAL_AVAILABLE and ground_truth is not None and current_model_answer is not None:
                 logger.info(f"Scoring task {task_id} attempt {current_attempt_num}...")
                 try:
                    # 使用从 response 获取的 current_model_answer 进行评分
                    is_correct = question_scorer(current_model_answer, ground_truth)
                    current_is_correct = is_correct # 记录当前尝试的正确性
                    final_result["is_correct"] = is_correct # 更新最终结果的正确性
                    logger.info(f"Real-time score for {task_id} attempt {current_attempt_num}: {'Correct' if is_correct else 'Incorrect'}")
                    if is_correct:
                        break # 成功则跳出重试循环
                 except Exception as score_err:
                      logger.error(f"Error scoring task {task_id} attempt {current_attempt_num}: {score_err}", exc_info=True)
                      current_is_correct = f"Scoring Error: {str(score_err)}" # 记录评分错误
                      final_result["is_correct"] = current_is_correct # 更新最终结果
            elif not api_error_occurred:
                 # 更新 is_correct 状态
                 if not EVAL_AVAILABLE: current_is_correct = "Scoring N/A (eval.py not imported)"
                 elif ground_truth is None: current_is_correct = "Scoring N/A (GT missing)"
                 elif current_model_answer is None: current_is_correct = False
                 final_result["is_correct"] = current_is_correct

        # --- 捕获 API/网络/解码 错误 ---
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out task {task_id} attempt {current_attempt_num}")
            current_error = "API request timed out"; api_error_occurred = True
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed task {task_id} attempt {current_attempt_num}: {e}")
            current_error = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None: current_status_code = e.response.status_code
            api_error_occurred = True
        except json.JSONDecodeError as e:
            response_text = response.text if 'response' in locals() else "N/A"
            status_code = current_status_code or final_result['api_response_status'] or 'N/A'
            logger.error(f"Failed to decode API JSON response task {task_id} attempt {current_attempt_num}. Status: {status_code}, Response: {response_text[:200]}... Error: {e}")
            current_error = f"Failed to decode API response (status {status_code})"
            api_error_occurred = True
        except Exception as e:
            logger.exception(f"Unexpected error task {task_id} attempt {current_attempt_num}: {e}")
            current_error = f"Unexpected error: {str(e)}"; api_error_occurred = True
        finally:
            duration = time.time() - start_time
            # 使用当前尝试的状态记录日志
            status_msg = "Success" if current_model_answer is not None and current_error is None else "Failed"
            correct_msg = f"Correct: {current_is_correct}" if current_is_correct is not None else "Correct: N/A"
            answer_preview = str(current_model_answer)[:100] + ('...' if current_model_answer and len(str(current_model_answer)) > 100 else '')
            # --- 修改日志输出, 确保打印当前尝试的信息 ---
            logger.info(
                f"Task {task_id} attempt {current_attempt_num} finished in {duration:.2f}s. "
                f"Status: {status_msg}. {correct_msg}. "
                f"Model Answer: '{answer_preview}'. " # 打印当前尝试答案预览
                f"Ground Truth: '{ground_truth}'. "
                f"Error: {current_error}" # 打印当前尝试错误
            )
            # --- 结束修改 ---

        # 更新最终结果中的错误状态
        if current_error:
            final_result["error"] = current_error
        if current_is_correct is False: # 如果当前尝试明确错误
             final_result["is_correct"] = False


        # --- 退出或继续重试 ---
        # 使用 current_is_correct 判断当前尝试是否成功
        if current_is_correct is True or api_error_occurred:
            logger.info(f"Exiting retry loop for task {task_id} after attempt {current_attempt_num}. Correct: {current_is_correct}, API Error: {api_error_occurred}")
            break # 成功或遇致命错误，退出循环
        elif attempt < MAX_RETRIES:
             logger.warning(f"Task {task_id} answer incorrect/error. Retrying ({current_attempt_num + 1}/{TOTAL_TRIES})...")
             time.sleep(2) # Optional delay
        else:
             logger.error(f"Task {task_id} still incorrect or error after {TOTAL_TRIES} attempts.")
             # 此时 final_result 中保存的是最后一次尝试的结果

    return final_result

# --- run_evaluation_script 函数 (保持不变) ---
def run_evaluation_script(results_filepath: str, metadata_filepath: str):
    logger.info(f"Attempting to run final evaluation using eval.py...")
    eval_script_path = os.path.join(os.path.dirname(__file__), "eval.py")
    if not os.path.exists(eval_script_path):
        logger.error(f"eval.py script not found at {eval_script_path}. Cannot run final evaluation.")
        return

    command = [
        sys.executable,
        eval_script_path,
        results_filepath,
        "--metadata_file", metadata_filepath
    ]
    try:
        logger.info(f"Executing command: {' '.join(command)}")
        process = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        logger.info("--- eval.py Output ---")
        print(process.stdout)
        if process.stderr:
            logger.warning("--- eval.py Stderr ---")
            print(process.stderr)
        logger.info("--- End of eval.py Output ---")
    except FileNotFoundError:
         logger.error(f"Error: Python executable '{sys.executable}' or eval script '{eval_script_path}' not found.")
    except subprocess.CalledProcessError as e:
        logger.error(f"eval.py script failed with return code {e.returncode}:")
        logger.error("--- eval.py Stdout ---")
        print(e.stdout)
        logger.error("--- eval.py Stderr ---")
        print(e.stderr)
    except Exception as e:
        logger.error(f"An unexpected error occurred while running eval.py: {e}", exc_info=True)

def main():
    """Main execution function."""
    logger.info(f"--- Starting GAIA Agent Runner ({GAIA_SPLIT}) ---")

    if not GAIA_SPLIT_DIR: logger.critical("GAIA validation data directory path not determined. Exiting."); return
    metadata_file = os.path.join(GAIA_SPLIT_DIR, "metadata.jsonl")
    gaia_metadata = load_gaia_metadata(metadata_file)
    if not gaia_metadata: logger.critical("No tasks loaded from metadata. Exiting."); return

    all_tasks = list(gaia_metadata.values())
    tasks_to_run: List[Dict[str, Any]] = []
    logger.info(f"Applying runner strategy: {RUNNER_STRATEGY}")
    # ... (选择 tasks_to_run 的逻辑不变) ...
    if RUNNER_STRATEGY == "single":
        if not RUNNER_TASK_ID: logger.critical("Strategy 'single' requires 'runner_task_id' in config. Exiting."); return
        single_task = gaia_metadata.get(RUNNER_TASK_ID)
        if not single_task: logger.error(f"Task ID '{RUNNER_TASK_ID}' not found. Exiting."); return
        tasks_to_run = [single_task]
        logger.info(f"Selected single task: {RUNNER_TASK_ID}")
    elif RUNNER_STRATEGY == "first_n":
        n = get_runner_first_n()
        if n is None or n <= 0: logger.critical("Strategy 'first_n' requires positive 'runner_first_n'. Exiting."); return
        tasks_to_run = all_tasks[:n]
        logger.info(f"Selected first {len(tasks_to_run)} tasks (requested: {n}).")
    else:
        tasks_to_run = all_tasks
        logger.info(f"Selected all {len(tasks_to_run)} tasks.")

    if not tasks_to_run: logger.warning("No tasks selected to run."); return


    try: # API Check
        logger.info(f"Pinging API server at {API_BASE_URL}...")
        ping_url = f"{API_BASE_URL}/docs"
        requests.get(ping_url, timeout=10).raise_for_status()
        logger.info(f"Successfully connected to API at {API_BASE_URL}")
    except requests.exceptions.RequestException as e:
        logger.critical(f"Error connecting to API at {API_BASE_URL} ({ping_url}): {e}. Is the server running?")
        return

    output_file_path = OUTPUT_FILE_TEMPLATE.format(timestamp=int(time.time()))
    processed_count = 0
    logger.info(f"Starting processing {len(tasks_to_run)} tasks with {MAX_WORKERS} worker(s). Max retries per task: {MAX_RETRIES}. Results -> {output_file_path}")

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(call_agent_api, task, f"{USER_ID_PREFIX}_{task['task_id']}", task.get("ground_truth")): task['task_id']
                for task in tasks_to_run
            }

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result() # 获取最终结果

                    # --- 修正：确保 is_correct 可序列化 ---
                    # 虽然 json.dumps 应该能处理 bool，但为了保险起见或排除其他可能性，
                    # 我们可以显式转换，或者更好地定义默认值
                    if isinstance(result.get("is_correct"), bool):
                        pass # Python bool is serializable by default json.dumps
                    elif result.get("is_correct") is None:
                         result["is_correct"] = None # None is fine
                    else:
                         # 如果是 'Scoring Error...' 或 'Scoring N/A...' 字符串，保持原样
                         # 如果是其他非 bool 类型，可能需要转换或记录错误
                         if not isinstance(result.get("is_correct"), str):
                              logger.warning(f"Non-standard type for is_correct in task {task_id}: {type(result.get('is_correct'))}. Converting to string.")
                              result["is_correct"] = str(result.get("is_correct"))
                    # --- 结束修正 ---

                    outfile.write(json.dumps(result, default=str) + '\n') # 添加 default=str 作为后备
                    outfile.flush()
                    processed_count += 1
                except Exception as e:
                    logger.exception(f"Critical error processing or writing result for task {task_id}: {e}")
                    # 记录错误到文件
                    error_result = {
                        "task_id": task_id, "model_answer": None,
                        "ground_truth": gaia_metadata.get(task_id, {}).get("ground_truth"),
                        "is_correct": False, "attempts": TOTAL_TRIES, # 假设尝试了所有次数
                        "error": f"Future processing/writing failed: {str(e)}",
                        "api_response_status": None,
                    }
                    try:
                        outfile.write(json.dumps(error_result) + '\n')
                        outfile.flush()
                    except Exception as write_err:
                         logger.error(f"Failed to write error result for task {task_id} to file: {write_err}")
                    processed_count += 1 # 即使写入失败，也算处理过

            logger.info("All submitted tasks futures completed.")

    except IOError as e: logger.critical(f"Error writing results to {output_file_path}: {e}"); return
    except Exception as e: logger.critical(f"An unexpected error occurred during task execution: {e}", exc_info=True)

    # --- Final Evaluation ---
    if processed_count > 0 and os.path.exists(output_file_path):
        logger.info(f"--- Running Final Evaluation on {output_file_path} ---")
        run_evaluation_script(output_file_path, metadata_file)
    else:
        logger.warning("No tasks were processed or results file not found, skipping final evaluation.")

    logger.info("--- GAIA Agent Runner Finished ---")

if __name__ == "__main__":
    if not GAIA_SPLIT_DIR:
        logger.critical("GAIA data directory path could not be determined. Check config.json. Exiting.")
    else:
        main()