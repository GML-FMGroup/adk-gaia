# src/tools/file_tools.py
import os
import json
import zipfile
from pptx import Presentation

try:
    # 尝试导入 pypdf，如果失败则尝试 PyPDF2
    import pypdf
    PdfReader = pypdf.PdfReader
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfReader
        print("Warning: pypdf not found, falling back to PyPDF2. Consider installing pypdf for better compatibility.")
    except ImportError:
        print("Error: Neither pypdf nor PyPDF2 found. PDF processing will fail. Please install one (`pip install pypdf` recommended).")
        PdfReader = None

from docx import Document as DocxDocument
from Bio.PDB import PDBParser
from typing import Dict, Any, Optional, Union, List
import io
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Google GenAI SDK Setup ---
try:
    from google import genai
    from google.genai import types as genai_types
    GENAI_SDK_AVAILABLE = True
    # 不再调用 genai.configure()
    # Client 实例将在需要时创建
    logger.info("google-genai SDK found.")
except ImportError:
    logger.error("google-genai SDK not installed. Multimodal features disabled.")
    genai = None # 保留 genai 为 None 以便后续检查
    genai_types = None
    GENAI_SDK_AVAILABLE = False


# --- Tool Implementations ---

def read_text_file(file_path: str) -> Dict[str, Any]:
    """Reads the content of a plain text file."""
    logger.info(f"Attempting to read text file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read text file: {file_path}")
        max_len = 10000
        if len(content) > max_len:
             logger.warning(f"Text file {file_path} truncated to {max_len} characters.")
             content = content[:max_len] + "\n... (truncated)"
        return {"status": "success", "content": content}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        return {"status": "error", "message": f"Error reading file: {str(e)}"}

def read_json_file(file_path: str) -> Dict[str, Any]:
    """Reads and parses a JSON or JSONL file."""
    logger.info(f"Attempting to read JSON/JSONL file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        logger.info(f"Successfully parsed JSON/JSONL file: {file_path}")
        content_str = json.dumps(data, indent=2)
        max_len = 10000
        if len(content_str) > max_len:
             logger.warning(f"JSON content from {file_path} truncated to {max_len} characters.")
             if isinstance(data, list):
                  content_str = json.dumps(data[:5], indent=2) + "\n... (list truncated)"
             elif isinstance(data, dict):
                   content_str = json.dumps(list(data.keys()), indent=2) + "\n... (object keys shown)"
             else:
                  content_str = content_str[:max_len] + "\n... (truncated)"

        return {"status": "success", "content": content_str}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return {"status": "error", "message": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return {"status": "error", "message": f"Error reading file: {str(e)}"}

def read_docx_file(file_path: str) -> Dict[str, Any]:
    """Reads text content from a DOCX file."""
    logger.info(f"Attempting to read DOCX file: {file_path}")
    try:
        document = DocxDocument(file_path)
        content = "\n".join([para.text for para in document.paragraphs])
        logger.info(f"Successfully read DOCX file: {file_path}")
        max_len = 10000
        if len(content) > max_len:
             logger.warning(f"DOCX content from {file_path} truncated.")
             content = content[:max_len] + "\n... (truncated)"
        return {"status": "success", "content": content}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error reading DOCX file {file_path}: {e}")
        return {"status": "error", "message": f"Error reading DOCX file: {str(e)}"}

def read_pptx_file(file_path: str) -> Dict[str, Any]:
    """Reads text content from a PowerPoint (.pptx) file."""
    logger.info(f"Attempting to read PPTX file: {file_path}")
    try:
        prs = Presentation(file_path)
        content_list = []
        for i, slide in enumerate(prs.slides):
            slide_text = [f"--- Slide {i+1} ---"]
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text.append(shape.text.strip())
            content_list.append("\n".join(filter(None, slide_text))) # 加入非空文本

        content = "\n\n".join(content_list)
        logger.info(f"Successfully extracted text from PPTX file: {file_path}")
        max_len = 15000 # PPTX 可能内容较多
        if len(content) > max_len:
            logger.warning(f"PPTX content from {file_path} truncated to {max_len} characters.")
            content = content[:max_len] + "\n... (truncated)"
        return {"status": "success", "content": content}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        # 捕捉 python-pptx 可能抛出的特定异常会更好
        logger.error(f"Error reading PPTX file {file_path}: {e}")
        return {"status": "error", "message": f"Error reading PPTX file: {str(e)}"}

def parse_pdb_file(file_path: str) -> Dict[str, Any]:
    """Parses a PDB file and returns a summary."""
    logger.info(f"Attempting to parse PDB file: {file_path}")
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", file_path)
        model = structure[0]
        summary = f"PDB Structure ID: {structure.id}\n"
        summary += f"Number of models: {len(structure)}\n"
        chain_ids = [chain.id for chain in model]
        summary += f"Chains in first model: {', '.join(chain_ids)}\n"
        # 更准确地计算原子数量
        atom_count = 0
        for chain in model:
            for residue in chain:
                 # 过滤掉 HETATM (非标准残基/水分子等) 如果需要
                 # if residue.id[0] == ' ': # Check if it's a standard residue
                 atom_count += len(residue)
        summary += f"Atom count in first model (excluding HETATM): {atom_count}\n" # 或包含 HETATM
        logger.info(f"Successfully parsed PDB file: {file_path}")
        return {"status": "success", "content": summary}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error parsing PDB file {file_path}: {e}")
        return {"status": "error", "message": f"Error parsing PDB file: {str(e)}"}

def extract_zip_content(file_path: str, target_filename: Optional[str] = None) -> Dict[str, Any]:
    """Lists contents of a ZIP file or extracts text from a specific file within it."""
    logger.info(f"Attempting to process ZIP file: {file_path}, target: {target_filename}")
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if target_filename:
                # 进行路径规范化和安全检查，防止路径遍历攻击
                normalized_target = os.path.normpath(target_filename).replace('\\', '/')
                if normalized_target.startswith('../') or normalized_target.startswith('/'):
                     logger.error(f"Invalid target filename (potential traversal): {target_filename}")
                     return {"status": "error", "message": "Invalid target filename."}

                if normalized_target in file_list:
                    try:
                        with zip_ref.open(normalized_target) as target_file:
                            file_bytes = target_file.read()
                            try:
                                content = file_bytes.decode('utf-8')
                            except UnicodeDecodeError:
                                logger.warning(f"Could not decode {target_filename} as UTF-8, trying latin-1.")
                                content = file_bytes.decode('latin-1', errors='replace')

                            logger.info(f"Successfully extracted text from {target_filename} in {file_path}")
                            max_len = 10000
                            if len(content) > max_len:
                                 logger.warning(f"Extracted content from {target_filename} truncated.")
                                 content = content[:max_len] + "\n... (truncated)"
                            return {"status": "success", "extracted_file": target_filename, "content": content}
                    except Exception as extract_err:
                        logger.error(f"Error extracting {target_filename} from {file_path}: {extract_err}")
                        return {"status": "error", "message": f"Could not extract or read {target_filename}: {str(extract_err)}"}
                else:
                    logger.warning(f"Target file {target_filename} (normalized: {normalized_target}) not found in {file_path}. Listing contents.")
                    return {"status": "success", "message": f"Target file '{target_filename}' not found.", "zip_contents": file_list}
            else:
                logger.info(f"Listing contents of ZIP file: {file_path}")
                return {"status": "success", "zip_contents": file_list}
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except zipfile.BadZipFile:
        logger.error(f"Invalid ZIP file: {file_path}")
        return {"status": "error", "message": "Invalid or corrupted ZIP file."}
    except Exception as e:
        logger.error(f"Error processing ZIP file {file_path}: {e}")
        return {"status": "error", "message": f"Error processing ZIP file: {str(e)}"}


# --- Tools using Gemini Native Multimodality ---

def process_pdf_with_gemini(file_path: str, prompt: str) -> Dict[str, Any]:
    """Processes a PDF file using the Gemini API."""
    logger.info(f"Attempting to process PDF with Gemini: {file_path}")
    if not GENAI_SDK_AVAILABLE: # 使用标志检查
        return {"status": "error", "message": "GenAI SDK not available."}
    try:
        # --- 创建 GenAI Client 实例 ---
        # 它会自动从环境变量读取 GOOGLE_API_KEY
        client = genai.Client()
        # -----------------------------

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()

        # 使用 genai_types
        pdf_part = genai_types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf')
        # 考虑从配置中获取模型名称
        model_name = "gemini-2.0-flash" # 应该选择支持文档处理的模型，如 gemini-1.5-flash 或更高版本

        # 检查模型支持 (可选但推荐)
        # try:
        #     model_info = client.models.get(f"models/{model_name}") # 使用 client.models.get
        #     if 'application/pdf' not in model_info.supported_content_mime_types:
        #          logger.error(f"Model {model_name} does not support PDF input.")
        #          return {"status": "error", "message": f"Model {model_name} does not support PDF input."}
        # except Exception as model_err:
        #      logger.warning(f"Could not verify model capabilities for {model_name}: {model_err}")


        logger.info(f"Sending PDF and prompt to Gemini model: {model_name}")
        # --- 使用 client 实例调用 API ---
        response = client.models.generate_content( # 或者 client.models.generate_content
             model=f"models/{model_name}", # 通常需要 'models/' 前缀
             contents=[pdf_part, prompt] # Pass file part and prompt
        )
        # -----------------------------

        logger.info(f"Received response from Gemini for PDF: {file_path}")
        return {"status": "success", "content": response.text}

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error processing PDF {file_path} with Gemini: {e}")
        return {"status": "error", "message": f"Error processing PDF with Gemini: {str(e)}"}


def process_audio_with_gemini(file_path: str, prompt: str) -> Dict[str, Any]:
    """Processes an audio file (e.g., MP3) using the Gemini API."""
    logger.info(f"Attempting to process audio with Gemini: {file_path}")
    if not GENAI_SDK_AVAILABLE: # 使用标志检查
        return {"status": "error", "message": "GenAI SDK not available."}

    # Determine MIME type
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".aiff": "audio/aiff",
        ".aac": "audio/aac", ".ogg": "audio/ogg", ".flac": "audio/flac",
        # 添加更多支持的格式
    }
    mime_type = mime_map.get(ext, "application/octet-stream")
    if mime_type == "application/octet-stream":
         logger.warning(f"Using generic MIME type for audio file: {file_path}")

    try:
        # --- 创建 GenAI Client 实例 ---
        client = genai.Client()
        # -----------------------------

        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        # 使用 genai_types
        audio_part = genai_types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
        # 考虑从配置获取模型
        model_name = "gemini-1.5-flash" # 或者 gemini-1.5-pro, 这些模型音频处理能力更强

        # 检查模型支持 (可选)
        # try:
        #     model_info = client.models.get(f"models/{model_name}")
        #     if mime_type not in model_info.supported_content_mime_types:
        #         logger.error(f"Model {model_name} does not support {mime_type} input.")
        #         return {"status": "error", "message": f"Model {model_name} does not support {mime_type} input."}
        # except Exception as model_err:
        #      logger.warning(f"Could not verify model capabilities for {model_name}: {model_err}")

        logger.info(f"Sending audio and prompt to Gemini model: {model_name}")
        # --- 使用 client 实例调用 API ---
        response = client.models.generate_content(
             model=f"models/{model_name}",
             contents=[audio_part, prompt]
        )
        # -----------------------------

        logger.info(f"Received response from Gemini for audio: {file_path}")
        return {"status": "success", "content": response.text}

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error processing audio {file_path} with Gemini: {e}")
        return {"status": "error", "message": f"Error processing audio with Gemini: {str(e)}"}

def process_image_with_gemini(file_path: str, prompt: str) -> Dict[str, Any]:
    """Processes an image file (e.g., PNG, JPEG) using the Gemini API."""
    logger.info(f"Attempting to process image with Gemini: {file_path}")
    if not GENAI_SDK_AVAILABLE:
        return {"status": "error", "message": "GenAI SDK not available."}

    # Determine MIME type
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".heic": "image/heic", ".heif": "image/heif",
        # 添加更多支持的格式
    }
    mime_type = mime_map.get(ext)
    if not mime_type:
         logger.error(f"Unsupported image file extension: {ext} for file {file_path}")
         return {"status": "error", "message": f"Unsupported image file type: {ext}"}

    try:
        client = genai.Client() # 创建客户端

        with open(file_path, "rb") as f:
            image_bytes = f.read()

        image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        # 考虑从配置获取模型，确保模型支持图像输入
        model_name = "gemini-1.5-flash" # 或者 gemini-1.5-pro / gemini-2.0-flash / gemini-2.5-flash/pro
        # model_name = "gemini-2.0-flash" # <-- 如果你的默认模型支持图像

        # 可选：检查模型能力
        # try:
        #     model_info = client.models.get(f"models/{model_name}")
        #     if mime_type not in model_info.supported_content_mime_types:
        #         logger.error(f"Model {model_name} does not support {mime_type} input.")
        #         return {"status": "error", "message": f"Model {model_name} does not support {mime_type} input."}
        # except Exception as model_err:
        #     logger.warning(f"Could not verify model capabilities for {model_name}: {model_err}")


        logger.info(f"Sending image and prompt to Gemini model: {model_name}")
        response = client.models.generate_content(
            model=f"models/{model_name}",
            contents=[image_part, prompt] # 图像在前，提示在后通常效果更好
        )

        logger.info(f"Received response from Gemini for image: {file_path}")
        return {"status": "success", "content": response.text}

    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return {"status": "error", "message": f"File not found: {file_path}"}
    except Exception as e:
        logger.error(f"Error processing image {file_path} with Gemini: {e}")
        return {"status": "error", "message": f"Error processing image with Gemini: {str(e)}"}