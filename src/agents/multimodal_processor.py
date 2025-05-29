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

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

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
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
         "专门处理和理解图像（如 PNG, JPG, JPEG, WEBP, HEIC, HEIF）和音频（如 MP3, WAV, AAC, OGG, FLAC）文件，"
        "利用 Gemini 的高级多模态AI能力。接收一个包含具体操作指令和文件绝对路径的 'request' 字符串。"
        "能够调用的工具包括：\n"
        "- `process_image_with_gemini`: 处理图像文件。需要参数: `file_path` (str, 图像文件的绝对路径), `prompt` (str, 对图像的操作指令，例如 '描述这张图片', '识别图中的物体', '图中的文字是什么？')。\n"
        "- `process_audio_with_gemini`: 处理音频文件。需要参数: `file_path` (str, 音频文件的绝对路径), `prompt` (str, 对音频的操作指令，例如 '转录此音频', '总结这段音频的主要内容', '这段音频的情绪是怎样的？')。\n"
        "协调器必须在 'request' 中提供清晰的指令和正确的绝对文件路径。智能体将根据文件类型选择合适的工具，并将指令作为 'prompt' 参数传递给相应的 Gemini 处理函数。"
    ),
    instruction=(
        "你是一位专业的多模态处理专家。你会收到一个名为 `request` 的字符串参数，"
        "其中包含指令以及一个指向图像或音频文件的绝对文件路径。\n"
        "**重要：** 你的任务是解析 `request` 字符串以提取文件路径和具体的操作/提示，然后调用相应的工具。\n"
        "1.  **解析请求：** 从输入的 `request` 字符串中提取**绝对文件路径**和**操作/提示**（例如，'描述这张图片'，'这张图片里是什么鸟类？'，'转录此音频'）。\n"
        "2.  **根据文件扩展名选择工具：**\n"
        "    - 对于图像文件（.png, .jpg, .jpeg, .webp, .heic, .heif），使用 `process_image_with_gemini`。\n"
        "    - 对于音频文件（.mp3, .wav, .aac, .ogg, .flac），使用 `process_audio_with_gemini`。\n"
        "3.  **执行工具：** 调用所选工具。将提取的**文件路径**作为 `file_path` 参数传递，并将提取的**操作/提示**作为工具的 `prompt` 参数传递。\n"
        "4.  **返回结果：** 传递工具输出中的 'content'（大语言模型的响应）或 'message'。"
    ),
    tools=[
        process_audio_tool,
        process_image_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"MultimodalProcessorAgent initialized with model: {MULTIMODAL_PROCESSOR_MODEL}")
logger.info(f"MultimodalProcessorAgent Tools: {[tool.name for tool in multimodal_processor_agent.tools]}")