import asyncio
from typing import Callable, List, Dict, Set
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup


# ==================================================
# Generic crawler core
# ==================================================

async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch HTML content of a page."""
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=20)) as response:
        response.raise_for_status()
        return await response.text()


def extract_links(html: str, base_url: str) -> Set[str]:
    """Extract all links from a page."""
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if href.startswith("#"):
            continue

        full_url = urljoin(base_url, href)
        links.add(full_url)

    return links


async def crawl(
    start_url: str,
    max_pages: int,
    link_filter: Callable[[str], bool],
    content_extractor: Callable[[str, str], Dict | None],
) -> List[Dict]:
    """
    Generic asynchronous crawler.

    Args:
        start_url: Initial URL to crawl.
        max_pages: Maximum number of pages to crawl.
        link_filter: Function deciding which links to follow.
        content_extractor: Function extracting structured content from HTML.

    Returns:
        List of extracted content dictionaries.
    """
    visited: Set[str] = set()
    queue: List[str] = [start_url]
    results: List[Dict] = []

    async with aiohttp.ClientSession() as session:
        while queue and len(visited) < max_pages:
            url = queue.pop(0)

            if url in visited:
                continue

            visited.add(url)

            try:
                html = await fetch_page(session, url)
            except Exception as exc:
                print(f"[WARN] Failed to fetch {url}: {exc}")
                continue

            # Extract structured content
            try:
                content = content_extractor(html, url)
                if content:
                    results.append(content)
            except Exception as exc:
                print(f"[WARN] Content extraction failed for {url}: {exc}")

            # Discover new links
            try:
                links = extract_links(html, url)
                for link in links:
                    if link_filter(link) and link not in visited:
                        queue.append(link)
            except Exception:
                pass

    return results


# ==================================================
# Pydantic AI documentation adapter
# ==================================================

def is_pydantic_ai_doc(url: str) -> bool:
    """Filter URLs belonging to Pydantic AI documentation."""
    parsed = urlparse(url)
    return (
        "pydantic.dev" in parsed.netloc
        and "/ai/" in parsed.path
    )


def extract_pydantic_ai_content(html: str, url: str) -> Dict | None:
    """Extract main documentation text from Pydantic AI docs."""
    soup = BeautifulSoup(html, "html.parser")

    main = soup.find("main")
    if not main:
        return None

    text = main.get_text(separator=" ", strip=True)

    if not text or len(text) < 200:
        return None

    return {
        "url": url,
        "content": text,
    }


# ==================================================
# Public crawl entry point
# ==================================================

async def crawl_pydantic_ai_docs(
    start_url: str,
    max_pages: int = 100,
) -> List[Dict]:
    """
    Crawl Pydantic AI documentation.

    Args:
        start_url: Entry URL for docs.
        max_pages: Maximum pages to crawl.

    Returns:
        List of documents with URL and extracted content.
    """
    return await crawl(
        start_url=start_url,
        max_pages=max_pages,
        link_filter=is_pydantic_ai_doc,
        content_extractor=extract_pydantic_ai_content,
    )


# ==================================================
# CLI / Local testing
# ==================================================

if __name__ == "__main__":
    START_URL = "https://docs.pydantic.dev/latest/ai/"

    docs = asyncio.run(
        crawl_pydantic_ai_docs(
            start_url=START_URL,
            max_pages=20,
        )
    )

    print(f"\nCrawled {len(docs)} pages\n")

    for doc in docs[:3]:
        print("=" * 80)
        print(doc["url"])
        print(doc["content"][:500], "...")
