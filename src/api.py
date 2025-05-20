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
   print("Please obtain an API key from Google AI Studio (https://aistudio.google.com/app/apikey) and update the .env file.")
   print("---------------\n")

# --- ADK Imports ---
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
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
    get_gaia_data_dir = lambda: None # Dummy function


# --- API Models ---
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.")
    session_id: Optional[str] = Field(None, description="Identifier for the chat session. If None, a new session is created.")
    task_id: str = Field(..., description="The GAIA task ID.")
    question: str = Field(..., description="The question from the GAIA task.")
    file_name: Optional[str] = Field(None, description="Optional file name associated with the task.")
    # state: Optional[Dict[str, Any]] = Field({}, description="Initial or existing session state.") # 考虑是否需要从外部传入 state

class ChatResponse(BaseModel):
    session_id: str
    model_answer: Optional[str] = None
    reasoning_trace: Optional[List[Dict[str, Any]]] = None # 或者直接用 Any
    error: Optional[str] = None

# --- FastAPI App ---
app = FastAPI(
    title="GAIA Solver Agent API",
    description="API endpoint to interact with the ADK-based GAIA solving agent system.",
)

# --- ADK Setup ---
session_service = InMemorySessionService()
APP_NAME = "gaia_solver_app" # 应用名称

# Runner 实例
runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Receives a GAIA task and returns the agent's final answer."""
    user_id = request.user_id
    task_id = request.task_id
    actual_session_id = request.session_id or str(uuid.uuid4())

    # --- Session Management ---
    try:
        session = session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=actual_session_id
        )
    except KeyError:
        session = None
    except Exception as e:
        logger.warning(f"Error retrieving session {actual_session_id}: {e}")
        session = None

    initial_state_for_session = {
        "current_task_id": task_id,
        "current_session_id_for_debug_log": actual_session_id  # 新增
    }
    logger.info(f"--- DEBUG: Session state being set/updated with: {initial_state_for_session} for session {actual_session_id} ---")

    if not session:
        logger.info(
            f"Creating new session: {actual_session_id} for user: {user_id} with state: {initial_state_for_session}")
        session = session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=actual_session_id,  # 使用 actual_session_id
            state=initial_state_for_session
        )
    else:
        logger.info(f"Using existing session: {actual_session_id}, updating state with: {initial_state_for_session}")
        session.state.update(initial_state_for_session)
        session_service.update_session(session)  # 保存更新后的会话

    # --- Prepare Input ---
    user_message = request.question
    content = Content(role='user', parts=[Part(text=user_message)])

    # --- Run Agent ---
    final_answer = None
    reasoning_events = []
    error_message = None

    try:
        logger.info(
            f"Running agent for session {actual_session_id}, task {task_id} with message: {user_message[:200]}...")
        async for event in runner.run_async(user_id=user_id, session_id=actual_session_id, new_message=content):
            try:
                # 尝试序列化事件，排除 None 值以减少冗余
                event_dict = event.model_dump(exclude_none=True, warnings=False)
            except Exception as dump_error:
                logger.warning(f"Could not serialize event {event.id} for task {task_id}. Error: {dump_error}")
                event_dict = {"event_id": event.id, "author": event.author, "error": "Serialization failed"}

            reasoning_events.append(event_dict)
            # logger.debug(f"Event received for task {task_id}: {json.dumps(event_dict, indent=2)}") # Debug log

            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    full_response = event.content.parts[0].text.strip()
                    logger.info(f"Raw final response for task {task_id}: {full_response}")
                    # 改进 FINAL ANSWER 提取逻辑
                    if "FINAL ANSWER:" in full_response:
                        # 从 "FINAL ANSWER:" 之后的部分开始提取
                        answer_part = full_response.split("FINAL ANSWER:", 1)[-1].strip()

                        # 处理可能的 Markdown 代码块或反引号
                        if answer_part.startswith("```") and answer_part.endswith("```"):
                            final_answer = answer_part[3:-3].strip()
                        elif answer_part.startswith("`") and answer_part.endswith("`"):
                            final_answer = answer_part[1:-1].strip()
                        # 增加对数字和简单字符串的常见清理
                        elif answer_part.replace('.', '', 1).isdigit() or answer_part.replace('-', '',
                                                                                              1).isdigit():  # 检查是否像数字
                            final_answer = answer_part.replace(',', '')  # 移除数字中的逗号
                        else:
                            final_answer = answer_part
                        logger.info(f"Extracted final answer for task {task_id}: '{final_answer}'")
                    else:
                        logger.warning(
                            f"Final response for task {task_id} did not contain 'FINAL ANSWER:'. Using full response.")
                        final_answer = full_response  # 回退到使用完整响应

        if final_answer is None and not error_message:
            logger.warning(f"Agent did not produce a discernible final answer for task {task_id}.")
            final_answer = "[Agent did not provide a final answer]"  # 提供一个明确的无答案标记


    except Exception as e:
        logger.exception(f"Error running agent for task {task_id}: {e}")
        error_message = f"An error occurred: {str(e)}"
        reasoning_events.append({"error": error_message, "traceback": traceback.format_exc()})  # 添加完整 traceback

        # --- Return Response ---
    return ChatResponse(
        session_id=actual_session_id,
        model_answer=final_answer,
        reasoning_trace=reasoning_events,
        error=error_message
    )

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "GAIA Solver Agent API is running."}

# --- Function for Uvicorn (if running directly) ---
# def run_api():
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     run_api()