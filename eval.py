# eval.py
import json
import os
import argparse
import logging
import re
import string
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 从 config.py 或直接定义 GAIA 数据目录 ---
try:
    from src.core.config import get_gaia_data_dir
    GAIA_BASE_DIR = get_gaia_data_dir()
except ImportError:
    logger.warning("Could not import config. Using default GAIA data directory './GAIA/2023'")
    GAIA_BASE_DIR = "./GAIA/2023" # 默认路径

GAIA_SPLIT = "validation" # 评估验证集
GAIA_SPLIT_DIR = os.path.join(GAIA_BASE_DIR, GAIA_SPLIT) if GAIA_BASE_DIR else None

# --- 评分函数 (保持不变) ---
def normalize_number_str(number_str: str) -> float:
    """Converts a string to a float after removing common units/commas."""
    if number_str is None: return float("inf")
    number_str = str(number_str)
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        match = re.match(r"([-+]?\d*\.?\d+)", number_str)
        if match:
            return float(match.group(1))
        else:
            return float(number_str)
    except ValueError:
        logger.warning(f"String '{number_str}' cannot be normalized to number str.")
        return float("inf")
    except Exception as e:
         logger.error(f"Unexpected error normalizing number string '{number_str}': {e}")
         return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    """Splits a string by any character in char_list."""
    if s is None: return []
    s = str(s)
    pattern = f"[{''.join(re.escape(c) for c in char_list)}]"
    return [item.strip() for item in re.split(pattern, s)]


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalizes a string by removing whitespace, punctuation (optional), and lowercasing."""
    if input_str is None: return ""
    input_str = str(input_str)
    no_spaces = re.sub(r"\s+", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        normalized = no_spaces.lower().translate(translator)
    else:
        normalized = no_spaces.lower()
    return normalized


def question_scorer(model_answer: Optional[str], ground_truth: Optional[str]) -> bool:
    """Scores the model answer against the ground truth based on GAIA rules."""
    def is_float(element: any) -> bool:
        try:
            float(str(element))
            return True
        except (ValueError, TypeError):
            return False

    if model_answer is None:
        # logger.info(f"Evaluating Model Answer: None | Ground Truth: {ground_truth} -> False") # 已在调用处记录
        return False
    if ground_truth is None:
        # logger.warning(f"Ground truth is None for model answer '{model_answer}'. Returning False.") # 已在调用处记录
        return False

    model_answer_str = str(model_answer)
    ground_truth_str = str(ground_truth)

    if is_float(ground_truth_str):
        # logger.info(f"Evaluating '{model_answer_str}' as a number (GT: {ground_truth_str}).")
        model_answer_cleaned = model_answer_str.strip()
        normalized_ma = normalize_number_str(model_answer_cleaned)
        normalized_gt = float(ground_truth_str)
        is_correct = np.isclose(normalized_ma, normalized_gt)
        logger.info(f"Normalized MA: {normalized_ma}, Normalized GT: {normalized_gt} -> Correct: {is_correct}")
        return bool(is_correct) # Ensure it's a Python bool

    elif any(char in ground_truth_str for char in [",", ";"]):
        # logger.info(f"Evaluating '{model_answer_str}' as a comma/semicolon separated list (GT: {ground_truth_str}).")
        gt_elems = split_string(ground_truth_str)
        ma_elems = split_string(model_answer_str)
        # logger.info(f"GT elements: {gt_elems} | MA elements: {ma_elems}")

        if len(gt_elems) != len(ma_elems):
            logger.warning(f"Answer lists have different lengths ({len(ma_elems)} vs {len(gt_elems)}), returning False.")
            return False

        comparisons = []
        for i, (ma_elem, gt_elem) in enumerate(zip(ma_elems, gt_elems)):
            item_correct = False
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                normalized_gt_elem = float(gt_elem)
                item_correct = np.isclose(normalized_ma_elem, normalized_gt_elem)
                # logger.info(f"  Item {i+1} (Number): MA='{ma_elem}'({normalized_ma_elem}), GT='{gt_elem}'({normalized_gt_elem}) -> {item_correct}")
            else:
                normalized_ma_elem = normalize_str(ma_elem, remove_punct=False)
                normalized_gt_elem = normalize_str(gt_elem, remove_punct=False)
                item_correct = (normalized_ma_elem == normalized_gt_elem)
                # logger.info(f"  Item {i+1} (String): MA='{ma_elem}'({normalized_ma_elem}), GT='{gt_elem}'({normalized_gt_elem}) -> {item_correct}")
            comparisons.append(item_correct)
        all_correct = all(comparisons)
        # logger.info(f"Overall list comparison result: {all_correct}")
        return all_correct
    else:
        # logger.info(f"Evaluating '{model_answer_str}' as a string (GT: {ground_truth_str}).")
        normalized_ma = normalize_str(model_answer_str, remove_punct=True)
        normalized_gt = normalize_str(ground_truth_str, remove_punct=True)
        is_correct = (normalized_ma == normalized_gt)
        logger.info(f"Normalized MA: '{normalized_ma}', Normalized GT: '{normalized_gt}' -> Correct: {is_correct}")
        return is_correct

# --- Helper Functions ---
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON Lines file."""
    # ... (代码同上一个版本) ...
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in {file_path}: {line.strip()}")
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.exception(f"Error reading JSONL file {file_path}: {e}")
        return []

# --- Main Evaluation Logic with Level Breakdown ---
def evaluate_results(results_file: str, metadata_file: str):
    """Evaluates the results against the ground truth and provides level-based statistics."""
    logger.info(f"Starting evaluation...")
    logger.info(f"Results file: {results_file}")
    logger.info(f"Metadata file: {metadata_file}")

    results_data = load_jsonl(results_file)
    metadata_list = load_jsonl(metadata_file) # Load metadata as list first

    if not results_data or not metadata_list:
        logger.error("Could not load results or metadata. Aborting evaluation.")
        return

    # Create dictionaries for quick lookup of ground truth and level by task_id
    ground_truths = {}
    task_levels = {}
    for task in metadata_list:
        task_id = task.get('task_id')
        if task_id:
            ground_truths[task_id] = task.get('Final Answer') or task.get('Final answer')
            task_levels[task_id] = task.get('Level') # 获取 Level 信息

    # Initialize counters for overall and per-level statistics
    overall_correct_count = 0
    overall_total_count = 0
    overall_missing_gt_count = 0

    level_stats = {
        1: {"total": 0, "correct": 0, "missing_gt": 0},
        2: {"total": 0, "correct": 0, "missing_gt": 0},
        3: {"total": 0, "correct": 0, "missing_gt": 0},
        "Unknown": {"total": 0, "correct": 0, "missing_gt": 0} # For tasks with no level info
    }
    evaluation_details = []

    # Iterate through the model results
    for result in results_data:
        task_id = result.get("task_id")
        model_answer = result.get("model_answer")
        # **修正：从 result 中获取 is_correct (如果 run_gaia.py 已计算) 或重新计算**
        is_correct_from_result = result.get("is_correct")


        if not task_id:
            logger.warning(f"Skipping result with missing task_id: {result}")
            continue

        overall_total_count += 1
        task_level = task_levels.get(task_id)
        current_level_key = task_level if task_level in [1, 2, 3] else "Unknown"
        level_stats[current_level_key]["total"] += 1

        ground_truth = ground_truths.get(task_id)

        final_is_correct = None # 用于最终判断

        if ground_truth is None:
            logger.warning(f"No ground truth found for task_id: {task_id}. Skipping evaluation for this task.")
            overall_missing_gt_count += 1
            level_stats[current_level_key]["missing_gt"] += 1
            # is_correct remains None
        else:
            if isinstance(is_correct_from_result, bool): # 如果 run_gaia.py 已经评测过且结果是布尔值
                final_is_correct = is_correct_from_result
                logger.info(f"\n--- Evaluating Task ID: {task_id} (Level: {task_level}, Pre-scored: {final_is_correct}) ---")
                if not final_is_correct: # 如果预评分为错，打印一下模型答案和GT
                     logger.info(f"Model Answer: '{model_answer}', Ground Truth: '{ground_truth}'")
            else: # 否则，重新评分
                logger.info(f"\n--- Evaluating Task ID: {task_id} (Level: {task_level}) ---")
                final_is_correct = question_scorer(model_answer, ground_truth)

            if final_is_correct is True:
                overall_correct_count += 1
                level_stats[current_level_key]["correct"] += 1

            logger.info(f"Result for Task ID {task_id}: {'Correct' if final_is_correct else 'Incorrect'}")
            logger.info(f"----------------------------------")

        evaluation_details.append({
            "task_id": task_id,
            "level": task_level,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "is_correct": final_is_correct, # 使用最终的判断
            "error_in_result": result.get("error")
        })


    # --- Print Summary ---
    logger.info("\n--- Overall Evaluation Summary ---")
    logger.info(f"Total results processed: {overall_total_count}")
    overall_evaluatable_count = overall_total_count - overall_missing_gt_count
    logger.info(f"Results with missing ground truth: {overall_missing_gt_count}")
    logger.info(f"Results evaluated: {overall_evaluatable_count}")
    logger.info(f"Total Correct answers: {overall_correct_count}")

    if overall_evaluatable_count > 0:
        overall_accuracy = (overall_correct_count / overall_evaluatable_count) * 100
        logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
    else:
        logger.info("Overall Accuracy: N/A (no results could be evaluated)")
    logger.info("----------------------------------")

    # --- Per-Level Statistics ---
    logger.info("\n--- Per-Level Evaluation Summary ---")
    total_metadata_tasks = len(metadata_list)
    for level in [1, 2, 3, "Unknown"]:
        stats = level_stats[level]
        level_total_in_metadata = sum(1 for t in metadata_list if (task_levels.get(t['task_id']) == level) or (level == "Unknown" and task_levels.get(t['task_id']) not in [1,2,3]))

        logger.info(f"Level {level}:")
        logger.info(f"  Tasks in metadata: {level_total_in_metadata} ({ (level_total_in_metadata / total_metadata_tasks) * 100 :.2f}% of total)")
        logger.info(f"  Tasks processed in results file: {stats['total']}")
        logger.info(f"  Tasks with missing ground truth: {stats['missing_gt']}")
        evaluatable_level = stats['total'] - stats['missing_gt']
        logger.info(f"  Tasks evaluated: {evaluatable_level}")
        logger.info(f"  Correct answers: {stats['correct']}")
        if evaluatable_level > 0:
            accuracy_level = (stats['correct'] / evaluatable_level) * 100
            logger.info(f"  Accuracy for Level {level}: {accuracy_level:.2f}%")
        else:
            logger.info(f"  Accuracy for Level {level}: N/A")
        logger.info("  --------------------------------")
    logger.info("----------------------------------")

    # 可选：将详细评估结果保存到文件
    # eval_output_file = results_file.replace(".jsonl", "_evaluation_details.jsonl")
    # try:
    #     with open(eval_output_file, 'w', encoding='utf-8') as f:
    #         for detail in evaluation_details:
    #             f.write(json.dumps(detail) + '\n')
    #     logger.info(f"Detailed evaluation results saved to: {eval_output_file}")
    # except IOError as e:
    #     logger.error(f"Could not write detailed evaluation results to {eval_output_file}: {e}")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GAIA agent results with level breakdown.")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the JSON Lines file containing the agent's results (e.g., gaia_validation_results_*.jsonl)."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=os.path.join(GAIA_SPLIT_DIR, "metadata.jsonl") if GAIA_SPLIT_DIR else None,
        help=f"Path to the GAIA metadata file containing ground truths and levels (default: attempts to find it in {GAIA_SPLIT_DIR})."
    )

    args = parser.parse_args()

    if not args.metadata_file:
         logger.critical(f"Metadata file path could not be determined. Please specify using --metadata_file or ensure GAIA_BASE_DIR is correct.")
    elif not os.path.exists(args.results_file):
        logger.critical(f"Results file not found: {args.results_file}")
    elif not os.path.exists(args.metadata_file):
         logger.critical(f"Metadata file not found: {args.metadata_file}")
    else:
        evaluate_results(args.results_file, args.metadata_file)