# src/agents/document_processor.py
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging

# 导入相关的工具函数
from src.tools.file_tools import (
    read_text_file,
    read_docx_file,
    read_pptx_file,
    process_pdf_with_gemini # Gemini PDF 处理
)

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
    description=(
        "Specializes in reading and extracting information from text-based documents "
        "such as TXT, PDF, DOCX, and PPTX. Can perform summarization, Q&A, and "
        "information extraction on these documents using Gemini for PDF processing."
    ),
    instruction=(
        "You are an expert document processor. You will receive a single string argument named `request` "
        "containing instructions and an absolute file path to a document.\n"
        "**IMPORTANT:** Your task is to parse the `request` string to extract the file path and the specific action requested, then call the appropriate tool.\n"
        "1.  **Parse Request:** Extract the **absolute file path** and the **action/prompt** (e.g., 'summarize this document', 'find mentions of X') from the input `request` string.\n"
        "2.  **Select Tool based on file extension:**\n"
        "    - For `.pdf` files, use `process_pdf_with_gemini`, passing the extracted action/prompt as the tool's `prompt` argument.\n"
        "    - For `.docx` files, use `read_docx_file`.\n"
        "    - For `.pptx` files, use `read_pptx_file`.\n"
        "    - For `.txt` or other plain text files, use `read_text_file`.\n"
        "3.  **Execute Tool:** Call the selected tool, passing the extracted **file path** as the `file_path` argument and the extracted action/prompt to `process_pdf_with_gemini` if used.\n"
        "4.  **Return Result:** Relay the 'content' or 'message' from the tool's output. If the tool was `process_pdf_with_gemini`, its output is the direct answer."
    ),
    tools=[
        read_text_tool,
        read_docx_tool,
        read_pptx_tool,
        process_pdf_tool,
    ],
)

logger.info(f"DocumentProcessorAgent initialized with model: {DOCUMENT_PROCESSOR_MODEL}")
logger.info(f"DocumentProcessorAgent Tools: {[tool.name for tool in document_processor_agent.tools]}")