from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
from litellm import AsyncOpenAI
import logfire
import asyncio
import httpx
import os

from utils import read_system_prompt
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from typing import List
from chromadb.api.models.Collection import Collection

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
model = OpenAIModel(os.getenv("LLM_MODEL_NAME"))

logfire.configure(send_to_logfire="if-token-present")

SYSTEM_PROMPT = read_system_prompt(os.getenv("SYSTEM_PROMPT_FILE"))

@dataclass
class PydanticAIDeps:
    collection: Collection
    openai_client: AsyncOpenAI

system_prompt = SYSTEM_PROMPT

""" system_prompt = 
You are a knowlegeable and courteous concierge for QCon London 2025 conference.
You have access to all information and details about the tracks, presentations, speaker bio, venue information, 
Your job is to help conference attendees to discover presentations or speakers that match their interest so they would have the best
experience from the conference.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Always make sure you look at the relevant documentation before answering unless you're certain about the answer.
Be honest when you can't find relevant information in the knowledge base.

When analyzing the content, make sure to:
1. Provide accurate information based on the stored content
2. Cite specific examples when possible
3. Be clear when you're making assumptions or inferring information

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
""" 

pydantic_ai_agent = Agent(
    model, system_prompt=system_prompt, deps_type=PydanticAIDeps, retries=2
)


async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL_NAME, input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536


@pydantic_ai_agent.tool
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], user_query: str
) -> str:
    """Retrieve relevant documentation chunks based on the query."""
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)

        results = ctx.deps.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"],
        )

        if not results["documents"][0]:
            return "No relevant documentation found."

        formatted_chunks = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            chunk_text = f"""
# {metadata['title']}

{doc}

Source: {metadata['url']}
"""
            formatted_chunks.append(chunk_text)

        return "\n\n---\n\n".join(formatted_chunks)

    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


@pydantic_ai_agent.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """Retrieve a list of all available documentation pages."""
    try:
        results = ctx.deps.collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return []

        urls = sorted(set(meta["url"] for meta in results["metadatas"]))
        return urls

    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []


@pydantic_ai_agent.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """Retrieve the full content of a specific documentation page."""
    try:
        results = ctx.deps.collection.get(
            where={"url": url}, include=["documents", "metadatas"]
        )

        if not results["documents"]:
            return f"No content found for URL: {url}"

        sorted_results = sorted(
            zip(results["documents"], results["metadatas"]),
            key=lambda x: x[1]["chunk_number"],
        )

        page_title = sorted_results[0][1]["title"].split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]

        for doc, _ in sorted_results:
            formatted_content.append(doc)

        return "\n\n".join(formatted_content)

    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
