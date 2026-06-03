from typing import List

from src.compaction.Compaction import Compaction
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.Message import Message, MessageRole

SUMMARIZATION_PROMPT = """You are a conversation summarizer.
Your task is to create a concise but complete summary of the conversation provided.
Preserve: key decisions, important facts, user intent, and any unresolved questions.
Respond ONLY with the summary text, no preamble."""


class Summarization(Compaction):
    threshold: int
    keep_last_n: int
    provider: BaseLLMProvider

    async def compact(self, messages: List[Message]) -> List[Message]:
        if len(messages) <= self.threshold:
            return messages

        print(f"Compacting {len(messages)} messages...")

        messages_to_summarize = [
            m for m in messages[: -self.keep_last_n] if m.role != MessageRole.SYSTEM
        ]
        recent_messages = messages[-self.keep_last_n :]
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]

        summarization_request = [
            Message(role=MessageRole.SYSTEM, content=SUMMARIZATION_PROMPT),
            *messages_to_summarize,
            Message(
                role=MessageRole.USER,
                content="Please summarize the conversation above.",
            ),
        ]

        response = await self.provider.chat(
            messages=summarization_request,
            tools=[],
            temperature=0.0,
        )

        summary_message = Message(
            role=MessageRole.ASSISTANT,
            content=f"[Conversation summary]: {response.content}",
        )

        return [*system_messages, summary_message, *recent_messages]
