# GAIA Solver Agent using Google ADK

## 简介

本项目旨在使用 Google 最新的 Agent Development Kit (ADK) Python SDK 构建一个多智能体（Multi-Agent）系统，以解决具有挑战性的 [GAIA 基准测试](https://huggingface.co/gaia-benchmark) 中的任务。GAIA 数据集包含需要复杂推理、多步骤规划和多样化工具（如网页搜索、文件处理、代码执行）才能解答的问题。

我们的目标是利用 ADK 的模块化和灵活性，创建一个能够有效分解任务、协作并最终在 GAIA 数据集上取得优异性能的 Agent 系统。

## 架构概述

本系统采用**分层多智能体架构**，其核心是一个**协调器 Agent (Orchestrator)**，负责接收任务、规划执行步骤并将具体子任务委托给一系列**专家 Agent (Specialists)**。

*   **GAIAOrchestratorAgent (协调器):**
    *   作为系统的入口和总指挥。
    *   使用强大的 LLM (如 `gemini-2.5-pro-preview-03-25`) 来理解 GAIA 任务（包括从问题文本中**提取文件路径**）。
    *   制定（隐式或显式的）执行计划。
    *   通过 ADK 的 `AgentTool` 机制将子任务委托给相应的专家 Agent。
    *   整合来自专家 Agent 的结果。
    *   按照 GAIA 要求格式化最终答案 (`FINAL ANSWER: ...`)。
*   **专家 Agents:**
    *   **WebResearcherAgent:** 负责执行网页搜索和内容提取。持有 `google_search` 工具和自定义的网页抓取工具 (待实现)。
    *   **CodeExecutorAgent:** 负责执行 Python 代码片段。持有 `built_in_code_execution` 工具。
    *   **FileProcessorAgent:** (核心文件处理单元) 负责处理各种格式的文件（txt, json, xlsx, csv, pdf, docx, pptx, mp3, wav, png, jpg, pdb, zip等）。它内部持有一系列 `FunctionTool`，一部分使用标准 Python 库（如 `pandas`, `pypdf`, `python-pptx`, `Biopython`），另一部分利用 `google-genai` SDK 实现对 PDF、音频和**图像**的原生多模态处理。**它直接接收文件路径作为工具参数。**
    *   **CalculatorAgent (未来):** 负责执行数学计算、统计和单位转换。
*   **Agent 间通信:** 主要通过 `AgentTool` 实现显式调用和结果返回。**Orchestrator Agent 在调用 FileProcessorAgent 时，会构造一个包含文件路径和具体操作指令的单一字符串参数 (`request`)。** `session.state` 可用于共享*其他*上下文信息（如果需要）。

## 项目结构

```
.
├── eval.py               # 根据GAIA官方评测函数评估指定jsonl文件Accurancy
├── .env                  # 存放 API 密钥
├── config.json           # 模型名称、端口、路径等配置
├── pyproject.toml        # 项目依赖管理 (使用 uv)
├── run_gaia.py           # 运行 GAIA 任务的主脚本
├── cli_chat.py           # 简单的命令行聊天客户端，用于测试
├── .gitignore            # Git 忽略文件配置
├── README.md             # 本文件
├── GAIA/                 # GAIA 数据集存放处 (需手动放置)
│   └── 2023/
│       ├── validation/
│       │   ├── metadata.jsonl
│       │   └── ... (附件)
│       └── test/         # (如果需要测试集)
│           ├── metadata.jsonl
│           └── ...
└── src/                  # 项目源代码
    ├── __init__.py
    ├── agents/           # Agent 定义
    │   ├── __init__.py
    │   ├── orchestrator.py
    │   ├── web_researcher.py
    │   ├── code_executor.py
    │   └── file_processor.py
    │   # ... (未来可能添加 calculator.py 等)
    ├── tools/            # 自定义工具实现
    │   ├── __init__.py
    │   └── file_tools.py
    │   # ... (未来可能添加 web_tools.py, calculation_tools.py 等)
    ├── api.py            # FastAPI 应用入口
    └── core/             # 核心工具或配置加载
        ├── __init__.py
        └── config.py       # 配置加载逻辑
```

## 安装与设置

### 1. 环境准备

*   **Python:** >= 3.9 (建议 3.10 或 3.11)
*   **Conda:** 用于环境管理（当然也可以使用 venv）。
*   **uv:** 用于快速安装依赖 ( `pip install uv` )。

### 2. 获取代码

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 3. 创建 Conda 环境

```bash
conda create --name gaia-adk-agent python=3.11 -y
conda activate gaia-adk-agent
```
*(将 `python=3.11` 替换为你希望使用的 Python 版本)*

### 4. 安装依赖

使用 `uv` 根据 `pyproject.toml` 安装所有必需的库（包括新增的 `python-pptx`）：

```bash
uv pip install .
```

### 5. 配置 API 密钥

*   复制或创建 `.env` 文件在项目根目录。
*   编辑 `.env` 文件，填入你的 `GOOGLE_API_KEY`。从 [Google AI Studio](https://aistudio.google.com/app/apikey) 获取。
*   确保 `GOOGLE_GENAI_USE_VERTEXAI` 设置为 `FALSE` (除非你打算使用 Vertex AI)。
*   **重要:** 将 `.env` 文件添加到你的 `.gitignore` 文件中，防止密钥泄露。

```dotenv
# .env
GOOGLE_API_KEY=AIzaSy...YOUR...KEY...HERE
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Optional: Vertex AI settings if GOOGLE_GENAI_USE_VERTEXAI=TRUE
# GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
# GOOGLE_CLOUD_LOCATION="your-gcp-region"
```

### 6. 放置 GAIA 数据集

*   下载 GAIA 数据集。
*   将其解压或放置到项目根目录下，结构如下（与 `config.json` 中的 `gaia_data_dir` 配置对应）：
    ```
    <项目根目录>/
        GAIA/
            2023/
                validation/
                    metadata.jsonl
                    <task_id>.pdf
                    <task_id>.xlsx
                    <task_id>.png
                    <task_id>.pptx
                    ...
                test/
                    ...
    ```

## 配置文件 (`config.json`)

此文件用于配置模型、API端口和运行参数：

*   `orchestrator_model`: 协调器 Agent 使用的 LLM。
*   `specialist_model_flash`/`pro`: 专家 Agent 使用的 LLM。
*   `gaia_data_dir`: GAIA 数据集根目录的相对路径。
*   `api_port`: FastAPI 服务器运行的端口。
*   `runner_strategy`: `run_gaia.py` 的运行策略 ("all", "single", "first_n")。
*   `runner_task_id`: 当 `runner_strategy` 为 "single" 时，指定要运行的任务 ID。
*   `runner_first_n`: 当 `runner_strategy` 为 "first_n" 时，指定要运行的前 N 个任务。

Tips: 可通过single策略针对gaia的单个问题进行测试以针对性的快速迭代一个类型的Agent或是Tool

## 运行 Agent

### 1. 启动 API 服务器

在**激活了 Conda 环境**的终端中，运行：

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port <端口号>
```

将 `<端口号>` 替换为 `config.json` 中 `api_port` 指定的端口（默认为9012，根据你的配置）。 `--reload` 选项用于开发，它会在代码更改时自动重启服务器。

### 2. 使用 CLI 聊天客户端 (用于快速测试)

API 服务器运行后，打开**另一个激活了 Conda 环境**的终端，运行：

```bash
python cli_chat.py
```

按照提示输入问题与 Agent 交互。如果需要 Agent 处理本地文件，**请在问题中明确提供文件的绝对路径**，例如："Summarize the document at /home/user/mydocs/report.pdf"。输入 `quit` 或 `exit` 退出。

## 运行 GAIA 评估 (`run_gaia.py`)

此脚本用于批量处理 GAIA 数据集中的任务，并将结果（包括模型答案和基础元数据）保存到 JSON Lines 文件中。

**重要:** 脚本现在会自动计算 GAIA 任务附件的绝对路径，并将其包含在发送给 Orchestrator Agent 的问题文本中。

1. **确保 API 服务器正在运行。**
2. **运行脚本:** 在**激活了 Conda 环境**的终端中，运行：
    ```bash
    python run_gaia.py
    ```
3. **查看结果:** 脚本会实时处理任务，并将每个任务的结果追加写入到根目录下名为 `gaia_<split>_results_<timestamp>.jsonl` 的文件中。运行结束后会打印总结信息。

## 添加新的 Agent 或 Tool (开发者指南)

### 添加新工具 (FunctionTool)

1.  **实现函数:** 在 `src/tools/` 下（例如 `src/tools/calculation_tools.py`）创建你的 Python 函数，确保它有清晰的类型提示和详细的 **docstring** (这对 LLM 理解工具至关重要)。函数的返回值最好是字典，包含 `status` 和 `content`/`message`。**如果工具需要处理文件，它应该接受一个 `file_path` 参数。**
2.  **包装工具:** 在需要使用该工具的 Agent 文件中（例如 `src/agents/calculator.py`），导入你的函数，并使用 `from google.adk.tools import FunctionTool` 将其包装：`my_new_tool = FunctionTool(func=your_tool_function)`。
3.  **注册工具:** 将创建的 `my_new_tool` 实例添加到对应 Agent 定义中的 `tools` 列表中。
4.  **更新指令:** 修改 Agent 的 `instruction`，告知 LLM 新工具的存在、功能以及何时应该调用它（包括需要传递哪些参数，如 `file_path`）。

### 添加新工具 (MCPTool)

[Model Context Protocol (MCP)](https://github.com/modelcontextprotocol/specification) 是一个开放标准，旨在规范 LLM 与外部工具、数据源和应用程序的交互。ADK 通过 `MCPToolset` 类支持与实现了 MCP 协议的服务端进行集成，让你的 Agent 可以使用这些外部 MCP 工具。

这适用于你需要让 ADK Agent 调用一个已经存在的、遵循 MCP 协议的服务（例如，文件系统服务、数据库连接器、特定的 API 包装器等）。

**步骤:**

1.  **识别 MCP 服务器信息:**
    *   **本地服务器 (Stdio):** 确定启动服务器所需的命令和参数（例如，`npx @modelcontextprotocol/server-filesystem /path/to/folder`）。可能还需要设置环境变量（如 API 密钥）。
    *   **远程服务器 (SSE):** 获取服务器的 URL 和任何必要的认证头信息。

2.  **异步加载 MCP 工具集:**
    *   使用 `google.adk.tools.mcp_tool.mcp_toolset.MCPToolset` 的 `from_server` *异步*方法来连接并获取工具。这通常需要在创建 Agent 之前在一个 `async` 函数中完成。
    *   你需要提供 `connection_params`，根据服务器类型选择 `StdioServerParameters` 或 `SseServerParams`。

3.  **管理生命周期 (`exit_stack`)**:
    *   `MCPToolset.from_server` 返回一个包含 `tools` 列表和 `exit_stack` 的元组。
    *   `exit_stack` (一个 `contextlib.AsyncExitStack` 实例) **极其重要**，它负责管理与 MCP 服务器的连接（特别是对于本地 Stdio 服务器，它管理着子进程）。
    *   你**必须**确保在应用程序退出或不再需要这些 MCP 工具时，调用 `await exit_stack.aclose()` 来正确关闭连接和清理资源。否则，可能会导致本地服务器进程变成僵尸进程或连接泄露。
    *   **注意:** 将 `exit_stack` 的管理整合到 FastAPI 应用的生命周期事件（`startup` 和 `shutdown`）或者 Agent `Runner` 的上下文中可能需要更高级的集成模式，这超出了简单添加到 Agent `tools` 列表的范畴。对于需要长时间运行的 Agent，如何优雅地处理 `exit_stack` 是一个需要仔细考虑的设计点。

4.  **注册工具到 Agent:**
    *   将从 `load_mcp_tools()` 获取的 `mcp_tools` 列表添加到你的 Agent 定义的 `tools` 参数中。

5.  **更新 Agent 指令:**
    *   修改 Agent 的 `instruction`，告知 LLM 新增的通过 MCP 提供的工具的功能和用法。`MCPToolset` 会尝试将 MCP 工具的描述传递给 ADK，但明确在指令中提及会更有帮助。

**重要考量:**

*   **异步初始化:** 加载 MCP 工具是一个异步过程，通常需要在事件循环中完成，这可能影响你组织 Agent 初始化代码的方式。
*   **生命周期管理:** 正确处理 `exit_stack` 对于资源的释放至关重要。在简单的脚本中，可以在脚本末尾关闭。在服务中（如 FastAPI），通常需要在应用的 `shutdown` 事件处理器中关闭。
*   **错误处理:** 需要添加健壮的错误处理逻辑，以应对 MCP 服务器无法启动、连接失败或工具调用错误等情况。
*   **性能:** 对于频繁调用的工具，启动本地 Stdio 服务器的开销可能比较大。远程 SSE 服务器可能更适合高并发场景，但需要部署和维护 MCP 服务器本身。

### 添加新专家 Agent

1.  **创建文件:** 在 `src/agents/` 目录下创建一个新的 Python 文件（例如 `calculator.py`）。
2.  **定义 Agent:** 在新文件中，导入 `LlmAgent` (或其他 Agent 类型)，定义你的新 Agent 实例，包括 `name`, `model`, `description`, `instruction` 和它所需的 `tools` (可以是 `FunctionTool`, `AgentTool`, 或内置工具)。
3.  **导出 Agent:** 在 `src/agents/__init__.py` 文件中导入并导出你的新 Agent 类或实例。
4.  **在协调器中注册:**
    *   在 `src/agents/orchestrator.py` 中，导入你的新 Agent 实例。
    *   使用 `from google.adk.tools import agent_tool` 将其包装成 `AgentTool`：`new_agent_tool = agent_tool.AgentTool(agent=your_new_agent_instance)`。
    *   将 `new_agent_tool` 添加到 `orchestrator_agent` 的 `tools` 列表中。
5.  **更新协调器指令:** 修改 `orchestrator_agent` 的 `instruction`，清楚地说明新专家 Agent 的能力以及**如何构造传递给它的参数**（例如，对于文件处理，是构造包含路径和指令的单一 `request` 字符串）。

## TODO / 未来工作

*   [ ] **完善工具集&&专家系统:**
    *   目前gaia验证集前100题结果已跑出，可根据结果进行针对性优化
    *   优化文件处理Agent，增强工具能力。考虑为FileProcessAgent添加本地代码解释器辅助文件处理
    *   优化WebSearchAgent，考虑集成 Playwright 用于动态网页，添加更健壮的网页内容提取（例如处理表格、特定标签）
    *   添加一些MCP工具，弥补少数题目没有还没有工具可用的尴尬境地
    *   实现 CalculatorAgent 及其工具
*   [ ] **添加重试机制:** 为 Google API 调用和可能的工具执行失败添加重试逻辑。目前相当一部分任务解决失败是由于谷歌API后端的500 Internal Error所致，添加重试应能显著提升准确率
*   [ ] **优化 Prompt Engineering:** 持续迭代和优化所有 Agent 的指令，提高任务理解、规划和工具使用能力，特别是最终答案格式的遵循。
*   [ ] **优化架构:** 考虑使用ADK官方的WebUI、CLI和API实现，之前没有注意到有这个。
*   [ ] **脱离GAIA使用:** 如果工具和Agent完善之后刷GAIA分数高的话，考虑基于此制作一个客户端，探寻Agent系统有趣的实际应用。
*   [ ] **添加可视化架构图:** 在 README 中加入更清晰的架构图。

## 贡献

欢迎贡献！如果您发现 Bug 或有改进建议，请提出 Issue 或 Pull Request。

## 许可证

本软件根据 Apache License, Version 2.0 ("许可证") 授权；除非符合许可证，否则您不得使用此文件。
您可以在以下网址获取许可证副本：

&nbsp;&nbsp;&nbsp;&nbsp; http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按"原样"基础提供的，**没有任何明示或暗示的担保或条件**。
请参阅许可证了解许可证下的特定语言管理权限和限制。