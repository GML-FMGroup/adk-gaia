# cli_chat.py
import requests
import json
import uuid
import os
from dotenv import load_dotenv
import time # 引入 time 模块

# --- 从配置加载器导入 ---
from src.core.config import get_api_port # <--- 导入 get_api_port

# --- 配置 ---
# 加载 .env 文件以确保 API key 等设置对本地测试有效（虽然主要由服务器使用）
load_dotenv()

API_HOST = os.getenv("API_HOST", "http://localhost") # <--- 使用 localhost
API_PORT = get_api_port() # <--- 从配置获取端口
API_BASE_URL = os.getenv("API_URL", f"{API_HOST}:{API_PORT}") # <--- 构建 URL
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
DEFAULT_USER_ID = f"cli_user_{uuid.uuid4()}"

def run_cli_chat():
    """启动交互式命令行聊天客户端。"""

    user_id = input(f"Enter your user ID (or press Enter for default: {DEFAULT_USER_ID}): ")
    if not user_id:
        user_id = DEFAULT_USER_ID
    print(f"Using User ID: {user_id}")

    # 为此 CLI 会话生成一个唯一的 session ID
    session_id = str(uuid.uuid4())
    print(f"\nConnecting to agent API at: {CHAT_ENDPOINT}")  # 告知用户连接地址
    print("Type your question or 'quit'/'exit' to end the session.")

    while True:
        try:
            question = input(f"\n[{user_id}] You: ")
            question_lower = question.lower().strip()

            if question_lower in ["quit", "exit"]:
                print("Exiting chat.")
                break

            if not question.strip():
                continue

            task_id = f"cli_task_{uuid.uuid4()}"

            request_payload = {
                "user_id": user_id,
                "session_id": session_id,
                "task_id": task_id,
                "question": question,
                "file_name": None
            }

            print("Sending request to agent...")
            start_time = time.time()
            try:
                response = requests.post(CHAT_ENDPOINT, json=request_payload, timeout=300) # 增加超时时间
                duration = time.time() - start_time

                if response.status_code == 200:
                    response_data = response.json()
                    model_answer = response_data.get("model_answer")
                    reason = response_data.get("reasoning_trace") # 获取 REASON
                    error = response_data.get("error")

                    print(f"\n[Agent] ({duration:.2f}s):") # 换行打印
                    if error:
                        print(f"  Error: {error}")
                    elif model_answer:
                        print(f"  FINAL_ANSWER: {model_answer}")
                        if reason:
                            print(f"  REASON: {reason}")
                    else:
                        print("  Agent returned an empty or unexpected response structure.")
                        if reason: # 即使答案为空，也可能想看原因
                            print(f"  REASON (if any): {reason}")


                else:
                    print(f"\nError from API (Status {response.status_code}): {response.text[:500]}") # 打印部分错误信息

            except requests.exceptions.RequestException as e:
                print(f"\nNetwork or API error connecting to {CHAT_ENDPOINT}: {e}")  # 明确错误连接
            except json.JSONDecodeError:
                print(
                    f"\nError: Could not decode API response (Status {response.status_code}) from {CHAT_ENDPOINT}: {response.text[:200]}")
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")

        except KeyboardInterrupt:
            print("\nExiting chat.")
            break

if __name__ == "__main__":
    print("--- GAIA Agent CLI Chat ---")
    run_cli_chat()