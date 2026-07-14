# agentic-cli

**agentic-cli** is a framework agnostic, modular conversational agent framework that connects a LLM with user-defined tools. The agent receives instructions, decides whether to call a tool, and displays the response to the user. It features a terminal UI built with [Rich](https://github.com/Textualize/rich).

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- One of the supported LLM providers:
  - [Ollama](https://ollama.com/) (optional) running locally with the desired model (e.g. `gemma4:e2b-mlx`)
  - [OpenRouter](https://openrouter.ai/) (optional) with a valid API key
- A [Tavily](https://tavily.com/) API key (free tier available) — used by the web search tool

## Installation

```bash
uv add -r requirements.txt
uv add --group dev -r requirements-dev.txt
```

## Configuration

Copy the environment file and add your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and set your keys:

```env
TAVILY_API_KEY=tvly-your-key-here
OPENROUTER_API_KEY=sk-or-your-key-here
```

Provider, model and other settings are configured in `config.json` at the project root:

```json
{
  "mode": "cli",
  "llm": {
    "provider": "openrouter",
    "model": "google/gemini-2.5-fl-exp-03-25:free",
    "base_url": "https://openrouter.ai/api",
    "api_key": "OPENROUTER_API_KEY",
    "retry_max_attempts": 3,
    "retry_base_delay": 1.0
  },
  "compaction": {
    "enabled": true,
    "strategy": "SLIDING_WINDOW",
    "sliding_window_size": 20,
    "summarization_threshold": 20,
    "summarization_keep": 6
  }
}
```

| Field | Description |
|---|---|
| `mode` | `"cli"` (default) or `"api"` |
| `llm.provider` | Provider name (`ollama` or `openrouter`) |
| `llm.model` | Model identifier for the chosen provider |
| `llm.base_url` | Base URL of the provider API |
| `llm.api_key` | Name of the environment variable that holds the API key (only needed for cloud providers) |
| `llm.retry_max_attempts` | Max retries on provider errors (timeout, 5xx) |
| `llm.retry_base_delay` | Base delay in seconds for exponential backoff |
| `compaction.strategy` | `NONE`, `SLIDING_WINDOW`, or `SUMMARIZATION` |
| `compaction.sliding_window_size` | Messages kept when using `SLIDING_WINDOW` |
| `compaction.summarization_threshold` | Messages before summarization kicks in |
| `compaction.summarization_keep` | Messages kept verbatim after summarization |

## Usage

The framework supports two modes:

### CLI mode (default)

```bash
uv run python -m src
```

Example session:

```
You: What's the weather in Paris?
Execute tool weather with input: {'city': 'Paris'}
Execute? (y/n): y
You: Can you list the files in the current directory?
Execute tool bash with input: {'command': 'ls -la'}
Execute? (y/n): y
```

Each time the agent decides to run a tool, it asks for confirmation before executing.

### API mode

Start the FastAPI server:

```bash
uv run python -m src api
```

Or set `"mode": "api"` in `config.json` and run:

```bash
uv run python -m src
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/chat` | Non-streaming chat — returns `{"result": "..."}` |
| `POST` | `/chat/stream` | Server-Sent Events (SSE) streaming chat |

Testing with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'

curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello!"}'
```

## Architecture

```
src/
├── __init__.py
├── __main__.py                        # Unified entry point — dispatches CLI/API mode
├── agent/
│   └── Agent.py                       # Core agent loop and streaming loop
├── api/
│   ├── api.py                         # FastAPI app with /chat and /chat/stream
│   └── schema/
│       └── ChatBody.py                # Request model: query + optional session_id
├── compaction/
│   ├── Compaction.py                  # Abstract base class for compaction strategies
│   ├── CompactionStrategy.py          # Enum: NONE, SLIDING_WINDOW, SUMMARIZATION
│   ├── CompactionRunner.py            # Dispatcher that selects and runs the strategy
│   └── strategies/
│       ├── SlidingWindow.py           # Keeps the last N messages
│       └── Summarization.py           # Summarizes older messages via the LLM
├── config/
│   ├── AppConfig.py                   # Global application configuration model
│   ├── CompactionConfig.py            # Compaction-specific settings
│   ├── LLMConfig.py                   # LLM provider settings
│   ├── MCPConfig.py                   # MCP server configuration
│   ├── MemoryConfig.py                # Memory settings
│   ├── ToolsConfig.py                 # Tool-specific configuration
│   └── UIConfig.py                    # UI settings (streaming, etc.)
├── llm/
│   ├── interfaces/
│   │   ├── BaseLLMProvider.py         # Abstract base class for LLM providers
│   │   └── StreamLLMChatResponse.py   # Streaming response model
│   ├── providers/
│   │   ├── __init__.py                # Factory: create_provider(config)
│   │   ├── openai_base/               # Shared base for OpenAI-compatible providers
│   │   │   ├── OpenAICompatibleProvider.py  # Pre-built chat/stream/format_messages
│   │   │   └── BaseOpenAIMessage.py         # Base TypedDicts for OpenAI format
│   │   ├── ollama/
│   │   │   └── OllamaProvider.py      # Extends OpenAICompatibleProvider
│   │   └── openrouter/
│   │       └── OpenRouterProvider.py  # Extends OpenAICompatibleProvider
│   ├── schema/
│   │   ├── Message.py
│   │   ├── ToolCall.py
│   │   ├── LLMChatResponse.py
│   │   └── Chat*Error.py
│   └── utils.py                       # retry_with_backoff helper
├── mcp_integration/
│   ├── __init__.py
│   ├── client.py                      # MCP client connection logic
│   ├── mcp_config.py                  # MCP configuration parsing
│   ├── mcp_registry.py                # MCP server discovery and tool loading
│   ├── mcp.json                       # MCP server configuration
│   └── utils.py                       # MCP utility helpers
├── memory/
│   ├── Session.py                     # Session data model
│   ├── SessionIndex.py                # Session index manager
│   ├── data/                          # Persistent session storage
│   ├── preamble.py                    # Session preamble generation
│   ├── summarize.py                   # Session summarization logic
│   └── utils.py                       # Memory utilities
├── shared/
│   ├── __init__.py
│   ├── config.py                      # JSON config loader
│   ├── console.py                     # Rich console helpers
│   ├── main.py                        # CLI entry point (agent loop)
│   └── utils.py                       # system_prompt builder, compaction dispatch
├── tools/
│   ├── __init__.py
│   ├── interfaces/
│   │   └── Tool.py                    # Abstract Tool + ToolResult
│   ├── registry.py                    # Central tool registry
│   ├── ToolRunner.py                  # Interactive executor with confirmation
│   └── tools/
│       ├── bash.py
│       ├── launch_subagent.py
│       ├── mcp_tool.py
│       ├── read_file.py
│       ├── recall.py
│       ├── weather.py
│       ├── web_search.py
│       └── write_file.py
└── config.json                        # User-facing configuration file
```

### Flow

1. The user writes a message.
2. The message list is **compacted** (summarized or truncated) to stay within the LLM context window.
3. The compacted messages are sent to the LLM along with the available tools.
4. If the LLM returns a tool call, confirmation is requested and then executed; the result is appended and the loop continues.
5. If the LLM returns text, it is displayed and control returns to the user.

## Compaction strategies

Compaction controls how the conversation history is reduced before each LLM call, preventing the context window from growing indefinitely.

| Strategy | Description |
|---|---|
| `NONE` | No compaction — messages pass through unchanged. |
| `SLIDING_WINDOW` | Keeps only the most recent 20 non-system messages; discards older ones. |
| `SUMMARIZATION` | Once the conversation exceeds 20 messages, summarizes older messages via the LLM and keeps the last 6 messages verbatim. |

The active strategy and its parameters are configured in `config.json` under the `compaction` key.

## Adding a LLM provider

Adding a new provider is plug and play — no core code needs to be modified beyond the provider package itself.

### Option A: OpenAI-compatible provider (recommended)

If your provider supports the OpenAI SDK (or exposes an OpenAI-compatible API), you can extend `OpenAICompatibleProvider`. This class already implements `format_messages`, `chat`, and `stream_chat` — you only need to pass `model`, `base_url`, and optionally `api_key` to the constructor.

1. **Create a package** under `src/llm/providers/<name>/` with a single file:
   - `XProvider.py` — a class that inherits from `OpenAICompatibleProvider`:

```python
from src.llm.providers.openai_base.OpenAICompatibleProvider import OpenAICompatibleProvider

class MyProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, base_url: str, api_key: str):
        super().__init__(model=model, base_url=base_url, api_key=api_key)
```

2. **Register it** in `src/llm/providers/__init__.py` (same as Option B, step 2).
3. **Configure it** via `config.json`.

This is what `OllamaProvider` and `OpenRouterProvider` do.

### Option B: Custom provider from scratch

If your provider does **not** support the OpenAI SDK, extend `BaseLLMProvider` directly.

1. **Create a package** under `src/llm/providers/<name>/` with at least two files:
   - `XProvider.py` — a class that inherits from `BaseLLMProvider` and implements:
     - `format_messages(messages)` — converts internal `Message` objects to the provider's native format
     - `chat(messages, tools, temperature)` — sends a non-streaming request
     - `stream_chat(messages, tools, temperature)` — sends a streaming request, yielding `StreamLLMChatResponse`
   - `XMessage.py` — TypedDict(s) describing the provider's message format

2. **Register the provider** in `src/llm/providers/__init__.py` by importing it and adding a new `if config.provider == "<name>"` branch in the `create_provider()` factory function.

3. **Configure it** via `config.json` — set `"provider"` to the new name, along with any needed `model`, `base_url`, and optionally `api_key` (the name of an env var).

That's it. The provider is automatically wired into the agent loop, tool system, memory, and compaction pipeline.

## Adding a tool

1. Create a file in `src/tools/tools/` with a class inheriting from `Tool`, defining `Args` (Pydantic) and `execute()`.
2. Import it and add it to the `TOOLS` list in `src/tools/registry.py`.

## CI

The project includes a GitHub Actions pipeline that runs:

- **Type checking**: `ty check`
- **Linting**: `ruff check`

Run locally:

```bash
uv run ty check
uv run ruff check
```

## Future work
- Skills
- Token count
- `POST /sessions`, `GET /sessions`, `GET /sessions/{id}`, `GET /tools` endpoints

## License

Check [LICENSE.md](LICENSE.md)
