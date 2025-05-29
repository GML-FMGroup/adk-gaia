# src/api.py
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import os
import logging
import traceback
import json  # <-- 导入 json 模块

# --- 配置日志 ---
logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
# 确保在导入任何使用环境变量的模块之前加载
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# 检查 GOOGLE_API_KEY 是否已设置
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_GOOGLE_API_KEY":
  print("\n--- WARNING ---")
  print("GOOGLE_API_KEY is not set or is still the placeholder in .env file.")
  print(
    "Please obtain an API key from Google AI Studio (https://aistudio.google.com/app/apikey) and update the .env file.")
  print("---------------\n")

# --- ADK Imports ---
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session  # <--- 修改导入路径
from google.genai.types import Content, Part

# --- 导入我们的 Orchestrator Agent ---
try:
  from src.agents import orchestrator_agent
  from src.core.config import get_gaia_data_dir
except ImportError as e:
  print(f"Error importing agents or config: {e}")
  print("Please ensure the project structure is correct and all __init__.py files exist.")
  # 提供一个假的 agent 以便 FastAPI 启动，但会报错
  from google.adk.agents import LlmAgent

  orchestrator_agent = LlmAgent(name="DummyAgent", model="gemini-2.0-flash", instruction="Error loading real agent.")
  get_gaia_data_dir = lambda: None  # Dummy function


# --- API Models ---
class ChatRequest(BaseModel):
  user_id: str = Field(..., description="Unique identifier for the user.")
  session_id: Optional[str] = Field(None,
                                    description="Identifier for the chat session. If None, a new session is created.")
  task_id: str = Field(..., description="The GAIA task ID.")
  question: str = Field(..., description="The question from the GAIA task.")
  file_name: Optional[str] = Field(None, description="Optional file name associated with the task.")


class ChatResponse(BaseModel):
  session_id: str
  model_answer: Optional[str] = None
  reasoning_trace: Optional[str] = None  # 修改为字符串以存储 Reason
  error: Optional[str] = None


# --- FastAPI App ---
app = FastAPI(
  title="GAIA Solver Agent API",
  description="API endpoint to interact with the ADK-based GAIA solving agent system.",
)

# --- ADK Setup ---
session_service = InMemorySessionService()
APP_NAME = "gaia_solver_app"

runner = Runner(
  agent=orchestrator_agent,
  app_name=APP_NAME,
  session_service=session_service,
)


# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
  user_id = request.user_id
  task_id = request.task_id
  actual_session_id = request.session_id or str(uuid.uuid4())

  # --- Session Management ---
  session: Optional[Session] = None  # 明确类型
  try:
    session = await session_service.get_session(
      app_name=APP_NAME, user_id=user_id, session_id=actual_session_id
    )
  except KeyError:
    logger.info(f"Session {actual_session_id} not found for user {user_id}. A new session will be created.")
    session = None
  except Exception as e:
    logger.warning(f"Error retrieving session {actual_session_id}: {e}")
    session = None

  initial_state_for_session = {
    "current_task_id": task_id,
    "current_session_id_for_debug_log": actual_session_id
  }
  logger.info(
    f"--- DEBUG: Session state being set/updated with: {initial_state_for_session} for session {actual_session_id} ---")

  if not session:
    logger.info(
      f"Creating new session: {actual_session_id} for user: {user_id} with state: {initial_state_for_session}")
    session = await session_service.create_session(
      app_name=APP_NAME,
      user_id=user_id,
      session_id=actual_session_id,
      state=initial_state_for_session
    )
  else:
    logger.info(f"Using existing session: {actual_session_id}, updating state with: {initial_state_for_session}")
    if session:  # 确保 session 不是 None
      session.state.update(initial_state_for_session)  # 使用 update 方法更新字典
      await session_service.update_session(session)  # 确保会话服务有 update_session 方法或正确的保存机制
    else:  # 理论上不应该发生，因为如果 session 是 None，上面会创建新的
      logger.error(
        f"Session object was None before attempting to update state for {actual_session_id}. This should not happen.")
      # 可以考虑重新创建会话或抛出错误
      session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=actual_session_id,
        state=initial_state_for_session
      )

  user_message = request.question
  content = Content(role='user', parts=[Part(text=user_message)])

  final_answer_str = None
  reason_text = None
  error_message = None

  try:
    logger.info(
      f"Running agent for session {actual_session_id}, task {task_id} with message: {user_message[:200]}...")
    async for event in runner.run_async(user_id=user_id, session_id=actual_session_id, new_message=content):
      pass

    final_orchestrator_event_text = None
    temp_events_list = []  # 用于临时存储事件以找到最后一个
    async for event in runner.run_async(user_id=user_id, session_id=actual_session_id, new_message=content):
      temp_events_list.append(event)  # 收集所有事件

    if temp_events_list:
      for event_obj in reversed(temp_events_list):  # 从后往前找
        if event_obj.author == orchestrator_agent.name and \
            event_obj.content and event_obj.content.parts and \
            event_obj.content.parts[0].text and \
            event_obj.is_final_response():  # 确保是最终响应的一部分
          final_orchestrator_event_text = event_obj.content.parts[0].text.strip()
          logger.info(
            f"Raw final event text from orchestrator for task {task_id}: {final_orchestrator_event_text[:500]}")
          break

    if final_orchestrator_event_text:
      try:
        # 清理可能的Markdown代码块标记
        cleaned_json_text = final_orchestrator_event_text
        if cleaned_json_text.startswith("```json"):
          cleaned_json_text = cleaned_json_text[len("```json"):].strip()
        if cleaned_json_text.startswith("```"):
          cleaned_json_text = cleaned_json_text[len("```"):].strip()
        if cleaned_json_text.endswith("```"):
          cleaned_json_text = cleaned_json_text[:-len("```")].strip()

        parsed_json = json.loads(cleaned_json_text)
        final_answer_str = parsed_json.get("FINAL_ANSWER")
        reason_text = parsed_json.get("REASON")
        logger.info(f"Parsed FINAL_ANSWER for task {task_id}: '{final_answer_str}'")
        logger.info(f"Parsed REASON for task {task_id}: '{reason_text[:200]}...'")

        if not isinstance(final_answer_str, str) and final_answer_str is not None:
          final_answer_str = str(final_answer_str)  # 确保是字符串
        if not isinstance(reason_text, str) and reason_text is not None:
          reason_text = str(reason_text)


      except json.JSONDecodeError as e:
        logger.error(
          f"Failed to decode JSON from orchestrator output for task {task_id}. Output: '{final_orchestrator_event_text[:500]}'. Error: {e}")
        error_message = "Agent output was not valid JSON."
        final_answer_str = f"[Agent output non-JSON: {final_orchestrator_event_text[:100]}]"  # 记录原始输出片段
        reason_text = "Failed to parse agent's JSON output."
      except Exception as e_parse:
        logger.error(
          f"Error parsing orchestrator JSON output for task {task_id}: {e_parse}. Output: {final_orchestrator_event_text[:500]}")
        error_message = f"Error parsing agent JSON: {str(e_parse)}"
        final_answer_str = f"[Error parsing agent JSON: {final_orchestrator_event_text[:100]}]"
        reason_text = f"Error during JSON parsing: {str(e_parse)}"

    if final_answer_str is None and not error_message:
      logger.warning(f"Agent did not produce a discernible final answer for task {task_id}.")
      final_answer_str = "[Agent did not provide a final answer]"
      reason_text = reason_text or "Agent did not provide a reason or final answer was missing."


  except Exception as e:
    logger.exception(f"Error running agent for task {task_id}: {e}")
    error_message = f"An error occurred: {str(e)}"
    if not reason_text:  # 如果还没有reason，用错误填充
      reason_text = f"System error during agent execution: {str(e)}"
    if not final_answer_str:
      final_answer_str = "[System error during execution]"

  return ChatResponse(
    session_id=actual_session_id,
    model_answer=final_answer_str,
    reasoning_trace=reason_text,  # 返回提取的 Reason
    error=error_message
  )


@app.get("/")
async def read_root():
  return {"message": "GAIA Solver Agent API is running."}