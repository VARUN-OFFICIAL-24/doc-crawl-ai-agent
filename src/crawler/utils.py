import asyncio
from urllib.parse import urlparse, urlunparse
from typing import Iterable, List


# ==================================================
# URL utilities
# ==================================================

def normalize_url(url: str) -> str:
    """
    Normalize URLs by removing fragments and query noise.
    This prevents crawling the same page multiple times.
    """
    parsed = urlparse(url)
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/"),
            "",
            "",
            "",
        )
    )


def is_http_url(url: str) -> bool:
    """Check whether a URL is HTTP or HTTPS."""
    return url.startswith("http://") or url.startswith("https://")


# ==================================================
# Async helpers
# ==================================================

class AsyncSemaphore:
    """
    Lightweight async semaphore wrapper for rate limiting.
    """

    def __init__(self, limit: int):
        self._semaphore = asyncio.Semaphore(limit)

    async def __aenter__(self):
        await self._semaphore.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        self._semaphore.release()


async def gather_with_concurrency(
    limit: int,
    coroutines: Iterable,
) -> List:
    """
    Run coroutines with a maximum concurrency limit.

    Args:
        limit: Maximum number of concurrent tasks
        coroutines: Iterable of async callables

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(limit)

    async def sem_task(coro):
        async with semaphore:
            return await coro

    tasks = [asyncio.create_task(sem_task(c)) for c in coroutines]
    return await asyncio.gather(*tasks, return_exceptions=True)


# ==================================================
# Safety helpers
# ==================================================

def deduplicate_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []

    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def chunk_list(items: List, size: int) -> List[List]:
    """
    Split a list into fixed-size chunks.

    Useful for batching requests or embeddings.
    """
    return [items[i:i + size] for i in range(0, len(items), size)]
