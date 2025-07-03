# src/agents/document_processor.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging

# 导入相关的工具函数
from src.agents.document_processor.tools import (
    read_text_file,
    read_docx_file,
    read_pptx_file,
    process_pdf_with_gemini # Gemini PDF 处理
)

from src.core.debug_logger import adk_before_model_callback, adk_after_model_callback

logger = logging.getLogger(__name__)

DOCUMENT_PROCESSOR_MODEL = get_model("specialist_model_pro") # 使用 Pro 模型以获得更好的文档理解能力

if not DOCUMENT_PROCESSOR_MODEL:
    raise ValueError("Model for Document Processor Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
read_text_tool = FunctionTool(func=read_text_file)
read_docx_tool = FunctionTool(func=read_docx_file)
read_pptx_tool = FunctionTool(func=read_pptx_file)
process_pdf_tool = FunctionTool(func=process_pdf_with_gemini)

document_processor_agent = LlmAgent(
    name="DocumentProcessorAgent",
    model=DOCUMENT_PROCESSOR_MODEL,
    # description 和 instruction 先使用中文进行人工调试，稳定之后再使用LLM翻译为英文
    description=(
        "专门负责读取、解析和理解多种格式的文档内容，包括 TXT, DOCX, PPTX, 和 PDF。 "
        "接收一个包含任务指令和文档绝对文件路径的 'request' 字符串。 "
        "根据文件类型选择合适的工具：\n"
        "- `read_text_file`: 读取纯文本文档 (.txt)。需要参数: `file_path` (str)。\n"
        "- `read_docx_file`: 读取 Word 文档 (.docx)。需要参数: `file_path` (str)。\n"
        "- `read_pptx_file`: 读取 PowerPoint 演示文稿 (.pptx)，提取文本内容。需要参数: `file_path` (str)。\n"
        "- `process_pdf_with_gemini`: 使用 Gemini API 处理 PDF 文档，可执行摘要、问答、信息提取等。需要参数: `file_path` (str), `prompt` (str, 描述对PDF的具体操作)。\n"
        "协调器应确保 'request' 字符串中包含清晰的操作指令（特别是对于 PDF）和正确的绝对文件路径。"
    ),
    instruction=(
        "你是一位专业的文档处理专家。你会收到一个名为 `request` 的字符串参数，"
        "其中包含指令和一个指向文档的绝对文件路径。\n"
        "**重要：** 你的任务是解析 `request` 字符串以提取文件路径和请求的具体操作，然后调用相应的工具。\n"
        "1.  **解析请求：** 从输入的 `request` 字符串中提取**绝对文件路径**和**操作/提示**（例如，'总结此文档'，'查找 X 的提及之处'）。\n"
        "2.  **根据文件扩展名选择工具：**\n"
        "    - 对于 `.pdf` 文件，使用 `process_pdf_with_gemini`，并将提取的操作/提示作为工具的 `prompt` 参数传递。\n"
        "    - 对于 `.docx` 文件，使用 `read_docx_file`。\n"
        "    - 对于 `.pptx` 文件，使用 `read_pptx_file`。\n"
        "    - 对于 `.txt` 或其他纯文本文件，使用 `read_text_file`。\n"
        "3.  **执行工具：** 调用所选工具，将提取的**文件路径**作为 `file_path` 参数传递，如果使用了 `process_pdf_with_gemini`，则将提取的操作/提示传递给它。\n"
        "4.  **返回结果：** 传递工具输出中的 'content' 或 'message'。如果使用的工具是 `process_pdf_with_gemini`，其输出即为直接答案。"
    ),
    tools=[
        read_text_tool,
        read_docx_tool,
        read_pptx_tool,
        process_pdf_tool,
    ],
    before_model_callback=adk_before_model_callback,
    after_model_callback=adk_after_model_callback
)

logger.info(f"DocumentProcessorAgent initialized with model: {DOCUMENT_PROCESSOR_MODEL}")
logger.info(f"DocumentProcessorAgent Tools: {[tool.name for tool in document_processor_agent.tools]}")