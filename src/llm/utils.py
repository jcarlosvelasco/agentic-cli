import asyncio
import random

from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, (ChatTimeoutError, ChatConnectionError)):
        return True
    if isinstance(e, ChatResponseError):
        return e.status_code is None or e.status_code >= 500
    return False


async def retry_with_backoff(
    coro_factory,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            if not _is_retryable(e) or attempt >= max_retries:
                raise
            last_exception = e
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, 0.1 * delay)
            await asyncio.sleep(delay + jitter)
    if last_exception is None:
        raise ChatResponseError("Retry failed after all attempts")
    raise last_exception
