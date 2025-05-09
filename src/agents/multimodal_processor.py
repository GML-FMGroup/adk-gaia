# src/agents/multimodal_processor.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging

# 导入相关的工具函数
from src.tools.file_tools import (
    process_audio_with_gemini,
    process_image_with_gemini
)

logger = logging.getLogger(__name__)

MULTIMODAL_PROCESSOR_MODEL = get_model("specialist_model_pro") # Pro 模型以获得更好的多模态能力

if not MULTIMODAL_PROCESSOR_MODEL:
    raise ValueError("Model for Multimodal Processor Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
process_audio_tool = FunctionTool(func=process_audio_with_gemini)
process_image_tool = FunctionTool(func=process_image_with_gemini)

multimodal_processor_agent = LlmAgent(
    name="MultimodalProcessorAgent",
    model=MULTIMODAL_PROCESSOR_MODEL,
    description=(
        "Specializes in processing and understanding image (PNG, JPG) and audio (MP3, WAV) files "
        "using advanced AI capabilities via Gemini. Can describe images, answer questions about them, "
        "transcribe audio, and understand audio content."
    ),
    instruction=(
        "You are an expert multimodal processor. You will receive a single string argument named `request` "
        "containing instructions and an absolute file path to an image or audio file.\n"
        "**IMPORTANT:** Your task is to parse the `request` string to extract the file path and the specific action/prompt, then call the appropriate tool.\n"
        "1.  **Parse Request:** Extract the **absolute file path** and the **action/prompt** (e.g., 'describe this image', 'what is the bird species in this picture?', 'transcribe this audio') from the input `request` string.\n"
        "2.  **Select Tool based on file extension:**\n"
        "    - For image files (.png, .jpg, .jpeg, .webp, .heic, .heif), use `process_image_with_gemini`.\n"
        "    - For audio files (.mp3, .wav, .aac, .ogg, .flac), use `process_audio_with_gemini`.\n"
        "3.  **Execute Tool:** Call the selected tool. Pass the extracted **file path** as the `file_path` argument and the extracted **action/prompt** as the tool's `prompt` argument.\n"
        "4.  **Return Result:** Relay the 'content' (the LLM's response) or 'message' from the tool's output."
    ),
    tools=[
        process_audio_tool,
        process_image_tool,
    ],
)

logger.info(f"MultimodalProcessorAgent initialized with model: {MULTIMODAL_PROCESSOR_MODEL}")
logger.info(f"MultimodalProcessorAgent Tools: {[tool.name for tool in multimodal_processor_agent.tools]}")