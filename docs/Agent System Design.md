# GAIA Solver Agent 系统设计（规划中）

本文档详细描述了为解决 GAIA 基准测试而设计的多智能体系统的架构、职能划分和工具配置。该系统利用 Google Agent Development Kit (ADK) 构建。

## 1. Agent 系统架构概述

本系统采用**分层多智能体架构**，其核心是一个**协调器 Agent (Orchestrator)**，负责接收任务、规划执行步骤并将具体子任务委托给一系列**专家 Agent (Specialists)**。

*   **GAIAOrchestratorAgent (协调器):**
    *   **核心职责:** 作为系统的唯一入口和大脑。接收原始 GAIA 问题，进行理解、拆解和规划。最关键的是，它需要识别任务所需的**能力类型**（网页搜索、文件处理、代码执行、计算等）以及**核心参数**（如搜索关键词、文件名、要执行的代码、计算公式等）。它还需要**准确提取和处理文件路径**，将相对路径（如果出现在问题中）转换为绝对路径传递给专家Agent。最终，它负责整合专家Agent返回的结果，并严格按照 GAIA 要求的格式 (`FINAL ANSWER: ...`) 输出最终答案。
    *   **模型:** 必须使用最强大的模型，如 `gemini-2.5-pro-preview-03-25`，以确保最佳的理解、推理和规划能力。

*   **专家 Agents:**
    *   **WebResearcherAgent (网页研究员):** 执行所有与互联网信息获取相关的任务。
    *   **CodeExecutorAgent (代码执行器):** 安全地执行代码片段，处理编程逻辑和特定库。
    *   **DocumentProcessorAgent (文档处理器):** 处理基于文本的文档格式（TXT, PDF, DOCX, PPTX）。
    *   **SpreadsheetDataAgent (电子表格数据分析师):** 处理表格数据文件（XLSX, CSV）。
    *   **MultimodalProcessorAgent (多模态处理器):** 处理图像、音频和视频文件。
    *   **SpecializedFileAgent (特殊文件处理器):** 处理 PDB, JSON-LD, ZIP 等特定文件格式。
    *   **CalculatorLogicAgent (计算与逻辑推理器):** 执行精确数学运算、逻辑推理和单位转换。

*   **Agent 间通信:** 主要通过 `AgentTool` 实现显式调用和结果返回。协调器 Agent 负责构造传递给专家 Agent 的参数（特别是包含文件路径和操作指令的单一 `request` 字符串）。`session.state` 可用于共享其他上下文信息（如果需要）。

## 2. Agent 职能与工具配置详解

下面详细列出了每个 Agent 的核心职责和建议配置的工具集。

### 2.1. GAIA Orchestrator Agent (协调器)

*   **职责:** 理解GAIA问题，规划步骤，提取文件路径，调用专家Agent，整合结果，格式化最终答案。
*   **工具:**
    *   `WebResearcherTool` (类型: `AgentTool`, 包装: `WebResearcherAgent`)
    *   `CodeExecutorTool` (类型: `AgentTool`, 包装: `CodeExecutorAgent`)
    *   `DocumentProcessorTool` (类型: `AgentTool`, 包装: `DocumentProcessorAgent`)
    *   `SpreadsheetDataTool` (类型: `AgentTool`, 包装: `SpreadsheetDataAgent`)
    *   `MultimodalProcessorTool` (类型: `AgentTool`, 包装: `MultimodalProcessorAgent`)
    *   `SpecializedFileTool` (类型: `AgentTool`, 包装: `SpecializedFileAgent`)
    *   `CalculatorLogicTool` (类型: `AgentTool`, 包装: `CalculatorLogicAgent`)
*   **说明:** Orchestrator 本身不直接执行外部操作，而是通过调用专家 Agent 的 `AgentTool` 来完成任务。其核心能力在于理解、规划、参数构造（特别是文件路径处理）和结果整合。

### 2.2. Web Researcher Agent (网页研究员)

*   **职责:** 执行网页搜索、特定网站信息提取（Wikipedia, arXiv, GitHub, 新闻, 数据库）、处理动态网页、访问网页存档 (Wayback Machine)。
*   **工具:**
    *   `google_search` (类型: ADK 内置) - 基础网页搜索。
    *   `web_scraper` (类型: `FunctionTool`, 实现: `requests` + `BeautifulSoup`) - 提取静态 HTML 内容 (文本、链接、表格)。*需要详细的 docstring 指导 URL 和提取目标。*
    *   `dynamic_web_scraper` (类型: `FunctionTool`, 实现: `playwright` 或 `selenium`) - 处理 JavaScript 渲染的动态网页。*需要参数指导 URL 和等待/交互。*
    *   `arxiv_search` (类型: `FunctionTool`, 实现: `arxiv` 库或 API) - 搜索 arXiv 论文（关键词、作者、日期、分类过滤）。
    *   `wikipedia_search` (类型: `FunctionTool`, 实现: `wikipedia` 库) - 获取 Wikipedia 页面内容或摘要。
    *   `github_inspector` (类型: `FunctionTool`, 实现: GitHub API 或 `PyGithub`) - 检查仓库 issues、提交历史、文件内容。
    *   `wayback_machine_checker` (类型: `FunctionTool`, 实现: Internet Archive API 或库) - 获取 URL 的历史存档版本。
    *   `tavily_search` / `serper_search` (类型: `LangchainTool` / `CrewaiTool`) - 备用/增强搜索引擎。*(需要 API Key)*

### 2.3. Code Executor Agent (代码执行器)

*   **职责:** 安全执行 Python 代码，处理编程逻辑、特定库（Biopython, statistics）、复杂计算、代码理解（Unlambda）。
*   **工具:**
    *   `built_in_code_execution` (类型: ADK 内置) - 执行 Python 代码 (需 Gemini 2+)。
    *   `python_interpreter` (类型: `FunctionTool`, 实现: 沙盒化 `subprocess`) - 备选或用于特定环境/库。**需极强安全措施。**
    *   `run_biopython_script` (类型: `FunctionTool`, 实现: 沙盒化 Biopython 执行) - 处理 PDB 文件等。
    *   `unlambda_interpreter` (类型: `FunctionTool`, 实现: 可能需集成解释器) - 处理 Unlambda。
    *   `cpp_compiler_runner` (类型: `FunctionTool`, 实现: 调用 g++/clang) - 处理 C++。**需极强安全措施。**

### 2.4. Document Processor Agent (文档处理器)

*   **职责:** 处理基于文本的文档格式（TXT, PDF, DOCX, PPTX），提取文本，查找信息，总结。
*   **工具:**
    *   `read_pdf_content` (类型: `FunctionTool`, 实现: 优先 `google-genai` 原生处理, 备选 `PyMuPDF`/`PyPDF2`) - 输入 `file_path`，输出文本内容/分析。
    *   `read_docx_content` (类型: `FunctionTool`, 实现: `python-docx`) - 输入 `file_path`，输出文本。
    *   `read_pptx_content` (类型: `FunctionTool`, 实现: `python-pptx`) - 输入 `file_path`，输出幻灯片文本。
    *   `read_text_file` (类型: `FunctionTool`, 实现: 标准 Python) - 输入 `file_path`，输出文本。
    *   `summarize_document` (类型: `FunctionTool`, 实现: 调用 LLM) - 输入文本，输出摘要。
    *   `find_info_in_document` (类型: `FunctionTool`, 实现: 文本提取 + 搜索/LLM) - 输入 `file_path`, `query`，输出相关信息。

### 2.5. Spreadsheet Data Agent (电子表格数据分析师)

*   **职责:** 处理表格数据文件（XLSX, CSV），执行数据读取、筛选、排序、计算、查找。
*   **工具:**
    *   `read_sheet_data` (类型: `FunctionTool`, 实现: `pandas`) - 输入 `file_path`, 可选 `sheet_name`，输出概览或数据。
    *   `get_cell_value` (类型: `FunctionTool`, 实现: `pandas`/`openpyxl`) - 输入 `file_path`, `sheet_name`, `cell_coordinate`，输出值。
    *   `query_spreadsheet` (类型: `FunctionTool`, 实现: `pandas`) - 输入 `file_path`, `sheet_name`, `query_string`，输出满足条件的行。
    *   `calculate_column_stat` (类型: `FunctionTool`, 实现: `pandas`) - 输入 `file_path`, `sheet_name`, `column_name`, `stat_type`，输出结果。
    *   `compare_rows_columns` (类型: `FunctionTool`, 实现: `pandas`) - 输入 `file_path`, `sheet_name`, `indices`, `comparison_type`，输出结果。

### 2.6. Multimodal Processor Agent (多模态处理器)

*   **职责:** 处理图像、音频、视频文件，进行内容理解、对象识别/边界框、OCR、音频转录、视频分析。
*   **工具:**
    *   `analyze_image_content` (类型: `FunctionTool`, 实现: `google-genai` SDK + 图像 `Part`) - 输入 `file_path`, `prompt`，输出分析。
    *   `get_image_bounding_boxes` (类型: `FunctionTool`, 实现: `google-genai` SDK + Prompt 要求 JSON 输出) - 输入 `file_path`, `object_description`，输出边界框。
    *   `transcribe_audio_file` (类型: `FunctionTool`, 实现: `google-genai` SDK + 音频 `Part`) - 输入 `file_path`, 可选 `prompt`，输出转录/分析。
    *   `analyze_video_content` (类型: `FunctionTool`, 实现: `google-genai` SDK + 视频 `Part` 或 YouTube URL) - 输入 `file_path`/`url`, `prompt`，输出分析。*(需 File API 处理大/长文件)*

### 2.7. Specialized File Agent (特殊文件处理器)

*   **职责:** 处理 PDB, JSON-LD, ZIP 等特定格式。
*   **工具:**
    *   `process_pdb_file` (类型: `FunctionTool`, 实现: `Biopython`) - 输入 `file_path`, `prompt`，执行特定分析。
    *   `extract_zip_contents` (类型: `FunctionTool`, 实现: `zipfile`) - 输入 `file_path`，输出列表或提取内容。
    *   `parse_jsonld` (类型: `FunctionTool`, 实现: `rdflib`/`jsonld`) - 输入 `file_path`, `prompt`，解析并查询 Linked Data。

### 2.8. Calculator & Logic Agent (计算与逻辑推理器)

*   **职责:** 执行精确数学运算、统计、单位转换、校验和计算、逻辑推理。
*   **工具:**
    *   `evaluate_mathematical_expression` (类型: `FunctionTool`, 实现: `sympy.sympify`/`numexpr`) - 输入表达式字符串，输出结果。**避免 `eval`。**
    *   `calculate_statistics` (类型: `FunctionTool`, 实现: `statistics`/`numpy`) - 输入数据列表, `stat_type`，输出结果。
    *   `unit_converter` (类型: `FunctionTool`, 实现: `pint`) - 输入数值、原始单位、目标单位，输出转换结果。
    *   `solve_logical_puzzle` (类型: `FunctionTool`, 实现: 特定逻辑或库) - 输入谜题描述/规则，输出解。
    *   `calculate_checksum` (类型: `FunctionTool`, 实现: 特定算法) - 输入序列, `algorithm_type`，输出校验位。
    *   `newtons_method_solver` (类型: `FunctionTool`, 实现: `scipy.optimize.newton`或手动) - 输入函数, 初始值，输出根。

---
**设计理念:**

*   **专业化:** 每个 Agent 专注于特定领域。
*   **分层控制:** Orchestrator 规划，专家 Agent 执行。
*   **显式调用:** 主要通过 `AgentTool`。
*   **覆盖性:** 旨在覆盖 GAIA 任务类型。
*   **鲁棒性:** 允许功能重叠。