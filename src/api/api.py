import asyncio
import json
from contextlib import asynccontextmanager
from os import path

from dotenv.main import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from src.agent.Agent import Agent
from src.api.schema.ChatBody import ChatBody
from src.llm.providers import create_provider
from src.mcp_integration.mcp_registry import MCPRegistry
from src.memory.Session import Session
from src.shared.config import load_config
from src.shared.utils import build_system_prompt, compact
from src.tools.registry import ToolRegistry

load_dotenv()
_config = load_config()
_provider = create_provider(_config.llm)
_session = Session()
_registry = ToolRegistry(provider=_provider, session=_session, config=_config)
_mcp_registry: MCPRegistry | None = None
_agent: Agent | None = None


async def setup():
    global _agent, _mcp_registry
    load_dotenv()

    if _config.mcp.enabled:
        mcp_path = _config.mcp.config_file
        if not path.exists(mcp_path):
            raise FileNotFoundError(f"No MCP config found at {mcp_path}")
        _mcp_registry = MCPRegistry.from_file(mcp_path)
        mcp_tools = await _mcp_registry.load_all()
        for tool in mcp_tools:
            _registry.register(tool)

    system_prompt = await build_system_prompt(_config)

    _agent = Agent(
        name="main",
        provider=_provider,
        tools=_registry.get_tools(),
        system_prompt=system_prompt,
        session=_session,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup()
    yield
    if _mcp_registry is not None:
        await _mcp_registry.cleanup()


app = FastAPI(lifespan=lifespan)


async def streaming_chat(query: str):
    agent = _agent
    if agent is None:
        raise RuntimeError("Agent not initialized")

    if _config.compaction.enabled:
        await compact(agent, _config.compaction.strategy, _provider, _config, True)

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def on_content(content: str):
        queue.put_nowait(json.dumps({"content": content}))

    async def run_agent():
        try:
            result, usage = await agent._stream_chat(query, on_content=on_content)
            await queue.put(
                json.dumps({"done": True, "content": result, "usage": usage})
            )
        except Exception as e:
            await queue.put(json.dumps({"error": str(e)}))
        finally:
            await queue.put(None)

    async def event_generator():
        task = asyncio.create_task(run_agent())
        while True:
            data = await queue.get()
            if data is None:
                break
            yield f"data: {data}\n\n"
        await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def agent_loop(query: str):
    if _agent is None:
        raise RuntimeError("Agent not initialized")

    if _config.compaction.enabled:
        await compact(_agent, _config.compaction.strategy, _provider, _config, True)

    result, usage = await _agent.chat(query)
    return result, usage


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chat")
async def chat(body: ChatBody):
    result, usage = await agent_loop(body.query)
    return {"result": result, "usage": usage}


@app.post("/chat/stream")
async def chat_stream(body: ChatBody):
    return await streaming_chat(body.query)
