# GAIA Solver Agent using Google ADK 🚀

## Introduction ℹ️

[GAIA](https://huggingface.co/gaia-benchmark) is made of more than 450 non-trivial question with an unambiguous answer, requiring different levels of tooling and autonomy to solve. It is therefore divided in 3 levels, where level 1 should be breakable by very good LLMs, and level 3 indicate a strong jump in model capabilities. Each level is divided into a fully public dev set for validation, and a test set with private answers and metadata.

This project is aim to use [ADK](https://google.github.io/adk-docs/) and [A2A](https://github.com/google/A2A) protocol to solve GAIA.

## Architecture 🏗️

The system employs a hierarchical multi-agent architecture where a central `OrchestratorAgent` analyzes incoming GAIA tasks and delegates sub-tasks to a suite of specialized agents, each equipped with specific tools and instructions for functions like web searching, document processing, code execution, or calculations. This division of labor allows each specialist agent to focus on its core competency, with the orchestrator coordinating their efforts to achieve the final solution.

## Start 🚀

### 1. Configure the environment through `pyproject.toml` ⚙️

### 2. Configure the API key 🔑

```dotenv
# .env
GOOGLE_API_KEY=AIzaSy...YOUR...KEY...HERE
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Optional: Vertex AI settings if GOOGLE_GENAI_USE_VERTEXAI=TRUE
# GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
# GOOGLE_CLOUD_LOCATION="your-gcp-region"
```
### 3. Configure `config.json` Or Keep default. 🛠️
*   **`orchestrator_model`**: LLM used by the main coordinating agent.
*   **`specialist_model_flash`**: LLM for fast, general-purpose specialist agents.
*   **`specialist_model_pro`**: LLM for more complex specialist agents (e.g., document/multimodal processing).
*   **`ollama_model`**: (Optional) Name of a locally run Ollama model.
*   **`gaia_data_dir`**: Path to the GAIA dataset directory.
*   **`api_port`**: Network port for the agent's API server.
*   **`runner_strategy`**: How to select tasks for `run_gaia.py` (`all`, `single`, `first_n`).
*   **`runner_task_id`**: Specific task ID to run if `runner_strategy` is `single`.
*   **`runner_first_n`**: Number of tasks to run if `runner_strategy` is `first_n`.
*   **`runner_max_retries`**: Maximum retry attempts for a failed task.
*   **`runner_max_workers`**: Number of concurrent tasks `run_gaia.py` can process.
*   **`gaia_split`**: Which GAIA dataset split to use (`validation` or `test`).
*   **`save_debug_chat_log`**: Boolean (`true`/`false`) to enable/disable saving detailed LLM interaction logs.

### 4. Run `uvicorn src.api:app --reload --host 0.0.0.0 --port 9012` 🌐

### 5. Run `python cli_chat.py` or `python run_gaia.py` 💻

## Performance / Todo 📊🎯

### Accuracy of GAIA (Latest) 📈
**Validation Set ( 2025-05-28 ):**
| Level   | Tasks Evaluated | Correct Answers | Accuracy (%) |
| :------ | :-------------- | :-------------- | :----------- |
| Level 1 | 53              | 40              | 75.47%       |
| Level 2 | 86              | 49              | 56.98%       |
| Level 3 | 26              | 11             | 42.31%       |
| **Overall** | **165**         | **100**          | **60.61%**   |

**Test Set ( 2025-05-10 ):**
* Accuracy: 37.54% * from GAIA official website

### Todo ✅
*   [ ] **Improve the prompts, agents and tools task by task** 🧠💡
*   [ ] **Use and build the client independently of GAIA** 🧩

## Contributing 🙌

Contributions are welcome! If you find a bug or have suggestions for improvement, please open an Issue or Pull Request.

## License 📜

This software is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

&nbsp;&nbsp;&nbsp;&nbsp; http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, **WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied**.
See the License for the specific language governing permissions and limitations under the License.
