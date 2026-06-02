# coding-agent

**coding-agent** is a minimal, modular conversational agent framework that connects a local LLM (Ollama) with user-defined tools. The agent receives instructions, decides whether to call a tool, and displays the response to the user.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with the desired model (e.g. `gemma4:e2b-mlx`)

## Installation

```bash
uv sync --dev
```

## Usage

```bash
uv run python src/shared/main.py
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

## Architecture

```
src/
├── llm/
│   ├── interfaces/BaseLLMProvider.py   # Abstract base class for LLM providers
│   ├── providers/ollama/               # Ollama implementation
│   │   ├── OllamaProvider.py
│   │   └── OllamaMessage.py
│   └── schema/                         # Data models
│       ├── Message.py
│       ├── ToolCall.py
│       ├── LLMChatResponse.py
│       └── Chat*Error.py
├── tools/
│   ├── interfaces/Tool.py              # Abstract Tool + ToolResult
│   ├── registry.py                     # Central tool registry
│   ├── ToolRunner.py                   # Interactive executor with confirmation
│   └── tools/
│       ├── bash.py                     # Execute bash commands
│       └── weather.py                  # Check weather via wttr.in
└── shared/main.py                      # Entry point (agent loop)
```

### Flow

1. The user writes a message.
2. It is sent to the LLM along with the available tools.
3. If the LLM returns a tool call, confirmation is requested and then executed.
4. If the LLM returns text, it is displayed as a response.

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

## License

MIT
