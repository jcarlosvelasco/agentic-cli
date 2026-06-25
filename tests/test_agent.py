from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.AppConfig import AppConfig
from src.agent.Agent import Agent
from src.llm.interfaces.StreamLLMChatResponse import StreamLLMChatResponse
from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall
from src.memory.Session import Session
from src.tools.interfaces.Tool import Tool, ToolResult
from tests.conftest import MockTool


def _make_config(**kwargs):
    cfg = AppConfig()
    for k, v in kwargs.items():
        setattr(cfg.tools, k, v)
    return cfg


@pytest.fixture
def session():
    return Session()


class TestAgentInit:
    def test_tools_by_name_returns_dict(self, mock_provider, mock_tool, session):
        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
        )
        assert agent.tools_by_name["mock_tool"] == mock_tool
        assert len(agent.tools_by_name) == 1

    def test_append_adds_to_both_messages_and_session(
        self, mock_provider, mock_tool, session
    ):
        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
        )
        msg = Message(role=MessageRole.USER, content="hello")
        agent.append(msg)
        assert len(agent.messages) == 1
        assert agent.messages[0] == msg
        assert len(session.messages) == 1
        assert session.messages[0] == msg


def _make_stream(chunks):
    async def stream(*args, **kwargs):
        for c in chunks:
            yield c

    return stream


def _make_stream_tool_then_text(tool_chunks, text_chunks):
    """Returns tool_chunks on first call, text_chunks on subsequent calls."""
    call_count = 0

    async def stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            for c in tool_chunks:
                yield c
        else:
            for c in text_chunks:
                yield c

    return stream


class TestAgentStreamLoop:
    @patch("src.agent.Agent.streaming_panel")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_text_response_returns_content(
        self,
        mock_display_result,
        mock_display_call,
        mock_streaming_panel,
        mock_provider,
        mock_tool,
        session,
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )

        mock_provider.stream_chat = _make_stream(
            [
                StreamLLMChatResponse(content="Hello", done=False),
                StreamLLMChatResponse(content=" World", done=False),
                StreamLLMChatResponse(content="", done=True),
            ]
        )

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            config=_make_config(confirm_execution=False),
        )

        result = await agent._stream_loop()

        assert result == "Hello World"
        assert len(agent.messages) == 1
        assert agent.messages[0].role == MessageRole.ASSISTANT
        assert agent.messages[0].content == "Hello World"

    @patch("src.agent.Agent.streaming_panel")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_tool_call_triggers_execution(
        self,
        mock_display_result,
        mock_display_call,
        mock_streaming_panel,
        mock_provider,
        mock_tool,
        session,
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )

        tool_call = ToolCall(id="call_1", name="mock_tool", args={"query": "test"})

        mock_provider.stream_chat = _make_stream_tool_then_text(
            tool_chunks=[
                StreamLLMChatResponse(content="", tool_calls=[tool_call], done=True)
            ],
            text_chunks=[StreamLLMChatResponse(content="done", done=True)],
        )

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            config=_make_config(confirm_execution=False),
        )
        agent.runner = AsyncMock()
        agent.runner.run.return_value = ToolResult(success=True, data="tool_output")

        result = await agent._stream_loop()

        assert result == "done"
        agent.runner.run.assert_awaited_once_with(
            mock_tool, tool_call.args, should_confirm=False
        )

    @patch("src.agent.Agent.streaming_panel")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_multiple_tool_calls_executed_concurrently(
        self,
        mock_display_result,
        mock_display_call,
        mock_streaming_panel,
        mock_provider,
        session,
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )

        tool1 = MockTool(name="tool1", result=ToolResult(success=True, data="res1"))
        tool2 = MockTool(name="tool2", result=ToolResult(success=True, data="res2"))

        calls = [
            ToolCall(id="c1", name="tool1", args={"a": 1}),
            ToolCall(id="c2", name="tool2", args={"b": 2}),
        ]

        mock_provider.stream_chat = _make_stream_tool_then_text(
            tool_chunks=[StreamLLMChatResponse(tool_calls=calls, done=True)],
            text_chunks=[StreamLLMChatResponse(content="done", done=True)],
        )

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[tool1, tool2],
            session=session,
            config=_make_config(confirm_execution=False),
        )
        agent.runner = AsyncMock()
        agent.runner.run.return_value = ToolResult(success=True, data="res")

        result = await agent._stream_loop()

        assert result == "done"
        assert agent.runner.run.await_count == 2

    @patch("src.agent.Agent.streaming_panel")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_unknown_tool_skipped_gracefully(
        self,
        mock_display_result,
        mock_display_call,
        mock_streaming_panel,
        mock_provider,
        session,
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )

        tool_call = ToolCall(id="c1", name="nonexistent_tool", args={})

        mock_provider.stream_chat = _make_stream_tool_then_text(
            tool_chunks=[StreamLLMChatResponse(tool_calls=[tool_call], done=True)],
            text_chunks=[StreamLLMChatResponse(content="done", done=True)],
        )

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[],
            session=session,
            config=_make_config(confirm_execution=False),
        )

        result = await agent._stream_loop()

        assert result == "done"
        tool_msgs = [m for m in agent.messages if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 0

    @patch("src.agent.Agent.streaming_panel")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_max_iterations_reached(
        self,
        mock_display_result,
        mock_display_call,
        mock_streaming_panel,
        mock_provider,
        session,
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )

        tool_call = ToolCall(id="c1", name="mock_tool", args={})

        mock_provider.stream_chat = _make_stream(
            [
                StreamLLMChatResponse(tool_calls=[tool_call], done=True),
            ]
        )

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "mock_tool"

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            max_iterations=2,
            config=_make_config(confirm_execution=False),
        )
        agent.runner = AsyncMock()
        agent.runner.run.return_value = ToolResult(success=True, data="output")

        result = await agent._stream_loop()
        assert result == "Max iterations reached"

    @patch("src.agent.Agent.streaming_panel")
    @patch("src.agent.Agent.confirm_execution")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_confirmation_pauses_and_resumes(
        self,
        mock_display_result,
        mock_display_call,
        mock_confirm,
        mock_streaming_panel,
        mock_provider,
        mock_tool,
        session,
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )
        mock_confirm.return_value = True

        tool_call = ToolCall(id="c1", name="mock_tool", args={})

        mock_provider.stream_chat = _make_stream_tool_then_text(
            tool_chunks=[StreamLLMChatResponse(tool_calls=[tool_call], done=True)],
            text_chunks=[StreamLLMChatResponse(content="done", done=True)],
        )

        cfg = AppConfig()
        cfg.tools.confirm_execution = True

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            config=cfg,
        )
        agent.runner = AsyncMock()
        agent.runner.run.return_value = ToolResult(success=True, data="output")

        await agent._stream_loop(on_content=lambda x: None, ui_control=mock_ctrl)

        mock_ctrl.pause.assert_called_once()
        mock_ctrl.resume.assert_called_once()
        mock_confirm.assert_called_once()

    @patch("src.agent.Agent.streaming_panel")
    async def test_stream_run_prepends_system_prompt(
        self, mock_streaming_panel, mock_provider, mock_tool, session
    ):
        mock_update = MagicMock()
        mock_ctrl = MagicMock()
        mock_streaming_panel.return_value.__aenter__.return_value = (
            mock_update,
            mock_ctrl,
        )

        mock_provider.stream_chat = _make_stream(
            [
                StreamLLMChatResponse(content="Done", done=True),
            ]
        )

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            system_prompt="You are helpful.",
            config=_make_config(confirm_execution=False),
        )

        result = await agent.stream_run("my task")
        assert result == "Done"
        assert agent.messages[0].role == MessageRole.SYSTEM
        assert agent.messages[0].content == "You are helpful."
        assert agent.messages[1].role == MessageRole.USER
        assert agent.messages[1].content == "my task"


class TestAgentLoop:
    @patch("src.agent.Agent.thinking_spinner")
    @patch("src.agent.Agent.display_assistant_message")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_loop_text_response(
        self,
        mock_display_result,
        mock_display_call,
        mock_display_assistant,
        mock_spinner,
        mock_provider,
        mock_tool,
        session,
    ):
        mock_spinner.return_value.__aenter__.return_value = None

        mock_provider.chat = AsyncMock()
        mock_provider.chat.return_value = LLMChatResponse(content="Hello World")

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            config=_make_config(confirm_execution=False),
        )

        result = await agent.run("my task")

        assert result == "Hello World"
        assert agent.messages[0].role == MessageRole.USER

    @patch("src.agent.Agent.thinking_spinner")
    @patch("src.agent.Agent.display_assistant_message")
    @patch("src.agent.Agent.display_tool_call")
    @patch("src.agent.Agent.display_tool_result")
    async def test_loop_tool_call_then_text(
        self,
        mock_display_result,
        mock_display_call,
        mock_display_assistant,
        mock_spinner,
        mock_provider,
        session,
    ):
        mock_spinner.return_value.__aenter__.return_value = None

        tool_call = ToolCall(id="c1", name="mock_tool", args={"x": 1})

        mock_provider.chat = AsyncMock()
        mock_provider.chat.side_effect = [
            LLMChatResponse(content="", tool_calls=[tool_call]),
            LLMChatResponse(content="Final answer"),
        ]

        mock_tool = MagicMock(spec=Tool)
        mock_tool.name = "mock_tool"

        agent = Agent(
            name="test",
            provider=mock_provider,
            tools=[mock_tool],
            session=session,
            config=_make_config(confirm_execution=False),
        )
        agent.runner = AsyncMock()
        agent.runner.run.return_value = ToolResult(success=True, data="tool_out")

        result = await agent.chat("do something")

        assert result == "Final answer"
        assert agent.runner.run.await_count == 1
        assert agent.messages[-1].role == MessageRole.ASSISTANT
