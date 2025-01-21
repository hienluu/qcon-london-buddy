# crawler.py
import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI

from vector_db import init_collection

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB collection
chroma_collection = init_collection()


@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif "\n\n" in chunk:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif ". " in chunk:
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)

    return chunks


async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from web content chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME"),
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}...",
                },
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
        }


async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME, input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)

    metadata = {
        "source": urlparse(url).netloc,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
    }

    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted["title"],
        summary=extracted["summary"],
        content=chunk,
        metadata=metadata,
        embedding=embedding,
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into ChromaDB."""
    try:
        chroma_collection.add(
            documents=[chunk.content],
            embeddings=[chunk.embedding],
            metadatas=[
                {
                    "url": chunk.url,
                    "chunk_number": chunk.chunk_number,
                    "title": chunk.title,
                    "summary": chunk.summary,
                    **chunk.metadata,
                }
            ],
            ids=[f"{chunk.url}_{chunk.chunk_number}"],
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting chunk: {e}")


async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    # Print number of URLs found
    print(f"Found {len(urls)} URLs to crawl")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)

        # Add a counter for progress tracking
        total_urls = len(urls)
        processed_urls = 0

        async def process_url(url: str):
            nonlocal processed_urls
            async with semaphore:
                result = await crawler.arun(
                    url=url, config=crawl_config, session_id="session1"
                )
                if result.success:
                    processed_urls += 1
                    print(
                        f"Successfully crawled: {url} ({processed_urls}/{total_urls})"
                    )
                    await process_and_store_document(
                        url, result.markdown_v2.raw_markdown
                    )
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")

        await asyncio.gather(*[process_url(url) for url in urls])
        print(f"Completed crawling {processed_urls} out of {total_urls} URLs")
    finally:
        await crawler.close()

def filter_urls(urls: List[str], pattern: str) -> List[str]:
    """filter URLs using the given pattern."""
    result = []
    
    for url in urls:
        if pattern in url:
            result.append(url)
    
    print(f"found {len(result)} urls with pattern: {pattern}")
    return result
    

def format_sitemap_url(url: str) -> str:
    """Format URL to ensure proper sitemap URL structure."""
    # Remove trailing slashes
    url = url.rstrip("/")

    # If URL doesn't end with sitemap.xml, append it
    if not url.endswith("sitemap.xml"):
        url = f"{url}/sitemap.xml"

    # Ensure proper URL structure
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    return url

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Get URLs from a sitemap."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()

        root = ElementTree.fromstring(response.content)
        namespace = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [loc.text for loc in root.findall(".//ns:loc", namespace)]

        print(f"Found {len(urls)} URLs in sitemap: {sitemap_url}")
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def crawl_site(url: str, additional_urls: List[str] = [], filter_str: str = "") -> List[str]:
    print(f"crawl_site => url: {url}, additional_urls: {len(additional_urls)}, filter_str: {filter_str}")
    
    formatted_url = format_sitemap_url(url)
    print(f"Processing {formatted_url}...")
    
    urls = get_urls_from_sitemap(formatted_url)
    
    if filter_str:
        urls = filter_urls(urls, filter_str)
    
    for url in additional_urls: 
        urls.append(url)
    
    print(f"Found {len(urls)} URLs in sitemap")
    
    #for url in urls:
    #    print(url)
    await crawl_parallel(urls)


if __name__ == "__main__":
     if len(sys.argv) > 1:
        print(f"len: {len(sys.argv)} Arguments:  {sys.argv[1:]}")
        if (len(sys.argv) == 2):
            scope = sys.argv[1]
            if scope == "site":
                site_url = "https://qconlondon.com"
                asyncio.run(crawl_site(site_url, ["https://qconlondon.com/faq/apr2025"], "apr2025"))
            elif scope == "url":
                url = sys.argv[1]
                print(f"url: {url}")
                #asyncio.run(crawl_parallel([url]))
     else:
        print("Usage: crawler.py [site|url]")
    #asyncio.run(main())
