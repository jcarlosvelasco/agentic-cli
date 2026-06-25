from src.config.CompactionConfig import CompactionConfig
from src.compaction.CompactionRunner import run_compaction
from src.compaction.CompactionStrategy import CompactionStrategy
from src.compaction.strategies.SlidingWindow import SlidingWindow
from src.compaction.strategies.Summarization import Summarization
from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message, MessageRole
from tests.conftest import MockProvider


class TestSlidingWindow:
    async def test_returns_all_messages_when_within_window(self, sample_messages):
        strategy = SlidingWindow(window_size=10)
        result = await strategy.compact(sample_messages)
        assert result == sample_messages

    async def test_truncates_non_system_when_exceeds_window(self, sample_messages):
        messages = sample_messages * 5
        strategy = SlidingWindow(window_size=5)
        result = await strategy.compact(messages)
        system_count = sum(1 for m in result if m.role == MessageRole.SYSTEM)
        non_system_count = sum(1 for m in result if m.role != MessageRole.SYSTEM)
        assert system_count == 5
        assert non_system_count == 5
        assert result[0].role == MessageRole.SYSTEM

    async def test_preserves_all_system_messages(self):
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys1"),
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.SYSTEM, content="sys2"),
            Message(role=MessageRole.USER, content="u2"),
            Message(role=MessageRole.USER, content="u3"),
            Message(role=MessageRole.USER, content="u4"),
        ]
        strategy = SlidingWindow(window_size=2)
        result = await strategy.compact(messages)
        system_msgs = [m for m in result if m.role == MessageRole.SYSTEM]
        non_system_msgs = [m for m in result if m.role != MessageRole.SYSTEM]
        assert len(system_msgs) == 2
        assert len(non_system_msgs) == 2
        assert non_system_msgs == [
            Message(role=MessageRole.USER, content="u3"),
            Message(role=MessageRole.USER, content="u4"),
        ]

    async def test_returns_empty_list_for_empty_input(self):
        strategy = SlidingWindow(window_size=10)
        result = await strategy.compact([])
        assert result == []

    async def test_window_size_of_zero_keeps_all_non_system(self):
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="u1"),
            Message(role=MessageRole.USER, content="u2"),
        ]
        strategy = SlidingWindow(window_size=0)
        result = await strategy.compact(messages)
        assert len(result) == 3
        assert result[0].role == MessageRole.SYSTEM

    async def test_only_system_messages_within_window_unchanged(self):
        messages = [
            Message(role=MessageRole.SYSTEM, content="sys1"),
            Message(role=MessageRole.SYSTEM, content="sys2"),
        ]
        strategy = SlidingWindow(window_size=10)
        result = await strategy.compact(messages)
        assert result == messages


class TestSummarization:
    async def test_returns_all_when_below_threshold(
        self, sample_messages, mock_provider
    ):
        strategy = Summarization(threshold=10, keep_last_n=3, provider=mock_provider)
        result = await strategy.compact(sample_messages)
        assert result == sample_messages

    async def test_summarizes_when_above_threshold(self):
        provider = MockProvider(
            chat_responses=[LLMChatResponse(content="Summary of conversation.")]
        )

        messages = [
            Message(role=MessageRole.USER, content=f"msg{i}") for i in range(25)
        ]
        sys_msg = Message(role=MessageRole.SYSTEM, content="be helpful")
        all_msgs = [sys_msg] + messages

        strategy = Summarization(threshold=10, keep_last_n=5, provider=provider)
        result = await strategy.compact(all_msgs)

        assert len(provider.chat_calls) == 1
        summarization_msgs = provider.chat_calls[0][0]
        assert any(
            "Please summarize the conversation above." in m.content
            for m in summarization_msgs
        )

        system_results = [m for m in result if m.role == MessageRole.SYSTEM]
        summary_results = [m for m in result if "Conversation summary" in m.content]
        recent_results = [
            m
            for m in result
            if m.role not in (MessageRole.SYSTEM, MessageRole.ASSISTANT)
        ]

        assert len(system_results) == 1
        assert len(summary_results) == 1
        assert len(recent_results) == 5

    async def test_preserves_system_messages_in_summarized_output(self):
        provider = MockProvider(
            chat_responses=[LLMChatResponse(content="Short summary.")]
        )

        messages = [
            Message(role=MessageRole.USER, content=f"msg{i}") for i in range(15)
        ]
        sys_msg = Message(role=MessageRole.SYSTEM, content="be helpful")
        all_msgs = [sys_msg] + messages

        strategy = Summarization(threshold=5, keep_last_n=3, provider=provider)
        result = await strategy.compact(all_msgs)

        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "be helpful"

    async def test_calls_provider_with_correct_temperature(self):
        provider = MockProvider(
            chat_responses=[LLMChatResponse(content="Short summary.")]
        )

        messages = [
            Message(role=MessageRole.USER, content=f"msg{i}") for i in range(15)
        ]

        strategy = Summarization(threshold=5, keep_last_n=3, provider=provider)
        await strategy.compact(messages)

        _, _, temperature = provider.chat_calls[0]
        assert temperature == 0.0

    async def test_handles_only_system_messages_above_threshold(self):
        provider = MockProvider(
            chat_responses=[LLMChatResponse(content="Short summary.")]
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content=f"sys{i}") for i in range(15)
        ]

        strategy = Summarization(threshold=5, keep_last_n=3, provider=provider)
        result = await strategy.compact(messages)

        assert len(result) > 0
        summary_msgs = [m for m in result if "Conversation summary" in m.content]
        assert len(summary_msgs) == 1

    async def test_summary_called_with_correct_messages(self):
        provider = MockProvider(chat_responses=[LLMChatResponse(content="Summary.")])

        messages = [
            Message(role=MessageRole.USER, content=f"normal_msg{i}") for i in range(15)
        ]

        strategy = Summarization(threshold=5, keep_last_n=3, provider=provider)
        await strategy.compact(messages)

        summarization_msgs = provider.chat_calls[0][0]
        assert len(summarization_msgs) == 2 + 12
        assert summarization_msgs[0].role == MessageRole.SYSTEM
        assert "summarizer" in summarization_msgs[0].content.lower()
        assert summarization_msgs[-1].role == MessageRole.USER
        assert "Please summarize" in summarization_msgs[-1].content


class TestCompactionRunner:
    async def test_sliding_window_strategy(
        self, sample_messages, mock_provider, config: CompactionConfig
    ):
        result = await run_compaction(
            CompactionStrategy.SLIDING_WINDOW, sample_messages, mock_provider, config
        )
        assert len(result) == len(sample_messages)

    async def test_none_strategy_returns_unchanged(
        self, sample_messages, mock_provider, config: CompactionConfig
    ):
        result = await run_compaction(
            CompactionStrategy.NONE, sample_messages, mock_provider, config
        )
        assert result == sample_messages

    async def test_summarization_strategy(self, config: CompactionConfig):
        provider = MockProvider(chat_responses=[LLMChatResponse(content="Summary.")])

        messages = [
            Message(role=MessageRole.USER, content=f"msg{i}") for i in range(25)
        ]
        result = await run_compaction(
            CompactionStrategy.SUMMARIZATION, messages, provider, config
        )
        assert len(result) < len(messages)
