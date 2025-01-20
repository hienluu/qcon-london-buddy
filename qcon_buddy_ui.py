from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
from datetime import datetime

import streamlit as st
import logfire
from openai import AsyncOpenAI


# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)
from pydantic_ai_agent import pydantic_ai_agent, PydanticAIDeps
from vector_db import get_chroma_client, init_collection
from crawler import crawl_parallel, get_urls_from_sitemap, filter_urls, format_sitemap_url

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB collection
chroma_collection = init_collection()

# Configure logfire to suppress warnings
logfire.configure(send_to_logfire="never")


class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""

    role: Literal["user", "model"]
    timestamp: str
    content: str



def get_db_stats():
    """Get statistics and information about the current database."""
    try:
        # Get all documents and metadata
        results = chroma_collection.get(include=["metadatas"])

        if not results["metadatas"]:
            return None

        # Get unique URLs
        urls = set(meta["url"] for meta in results["metadatas"])

        # Get domains/sources
        domains = set(meta["source"] for meta in results["metadatas"])

        # Get document count
        doc_count = len(results["ids"])

        # Format last updated time
        last_updated = max(meta.get("crawled_at", "") for meta in results["metadatas"])
        if last_updated:
            # Convert to local timezone
            dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            local_tz = datetime.now().astimezone().tzinfo
            dt = dt.astimezone(local_tz)
            last_updated = dt.strftime("%Y-%m-%d %H:%M:%S %Z")

        return {
            "urls": list(urls),
            "domains": list(domains),
            "doc_count": doc_count,
            "last_updated": last_updated,
        }
    except Exception as e:
        print(f"Error getting DB stats: {e}")
        return None


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "urls_processed" not in st.session_state:
        st.session_state.urls_processed = set()
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "current_progress" not in st.session_state:
        st.session_state.current_progress = 0
    if "total_urls" not in st.session_state:
        st.session_state.total_urls = 0


def initialize_with_existing_data():
    """Check for existing data and initialize session state accordingly."""
    stats = get_db_stats()
    if stats and stats["doc_count"] > 0:
        st.session_state.processing_complete = True
        st.session_state.urls_processed = set(stats["urls"])
        return stats
    return None


def display_message_part(part):
    """Display a single part of a message in the Streamlit UI."""
    if part.part_kind == "system-prompt":
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    """Run the agent with streaming text for the user_input prompt."""
    deps = PydanticAIDeps(collection=chroma_collection, openai_client=openai_client)

    async with pydantic_ai_agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        filtered_messages = [
            msg
            for msg in result.new_messages()
            if not (
                hasattr(msg, "parts")
                and any(part.part_kind == "user-prompt" for part in msg.parts)
            )
        ]
        st.session_state.messages.extend(filtered_messages)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )


async def process_url(url: str):
    """Process a single URL or sitemap URL."""
    try:
        progress_container = st.empty()
        with progress_container.container():
            # Format the URL
            formatted_url = format_sitemap_url(url)
            st.write(f"🔄 Processing {formatted_url}...")

            # First try as sitemap
            st.write("📑 Attempting to fetch sitemap...")
            urls = get_urls_from_sitemap(formatted_url)

            urls = filter_urls(urls, "apr2025")
            
            if urls:
                st.write(f"📎 Found {len(urls)} URLs in sitemap")
                # Create a progress bar
                progress_bar = st.progress(0, text="Processing URLs...")
                st.session_state.total_urls = len(urls)

                # Process URLs with status updates
                status_placeholder = st.empty()
                status_placeholder.text("⏳ Crawling web pages...")
                await crawl_parallel(urls)

                # Update status for post-processing steps
                status_placeholder.text("⚙️ Chunking documents...")
                await asyncio.sleep(0.1)  # Allow UI to update

                status_placeholder.text("🧮 Computing embeddings...")
                await asyncio.sleep(0.1)  # Allow UI to update

                status_placeholder.text("💾 Storing in database...")
                await asyncio.sleep(0.1)  # Allow UI to update

                progress_bar.progress(100, text="Processing complete!")
                status_placeholder.empty()  # Clear the status message
            else:
                # If sitemap fails, try processing as single URL
                st.write("❌ No sitemap found or empty sitemap.")
                st.write("🔍 Attempting to process as single URL...")
                original_url = url.rstrip(
                    "/sitemap.xml"
                )  # Remove sitemap.xml if it was added
                st.session_state.total_urls = 1

                status_placeholder = st.empty()
                status_placeholder.text("⏳ Crawling webpage...")
                await crawl_parallel([original_url])
                status_placeholder.empty()

            # Show summary of processed documents
            try:
                doc_count = len(chroma_collection.get()["ids"])
                st.success(
                    f"""
                ✅ Processing complete! 
                
                Documents in database: {doc_count}
                Last processed URL: {url}
                
                You can now start asking questions about the content.
                """
                )
            except Exception as e:
                st.error(f"Unable to get document count: {str(e)}")

    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")


async def main():
    st.set_page_config(
        page_title="QCon London Buddy", page_icon="🤖", layout="wide"
    )

    initialize_session_state()

    # Check for existing data
    existing_data = initialize_with_existing_data()

    st.markdown("<h3>QCon London Buddy - Chat Interface</h3>", unsafe_allow_html=True)
    

    if st.session_state.processing_complete:
        #st.subheader("Chat Interface")

        # Add suggested questions based on content
        with st.expander("📝 Suggested Questions", expanded=False):
            st.markdown(
                """
            Try asking:
            - "When is QCon London 2025 and where is it?"
            - "Tell me about all the 2025 tracks"
            - "What are the presentations in track AI and ML for Software Engineers?"
            - "Do you offer alumni discounts for returning QCon attendees?"
            """
            )

        # Display existing messages
        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                for part in msg.parts:
                    display_message_part(part)

        # Chat input
        user_input = st.chat_input("What would you like to know about QCon London 2025?")

        if user_input:
            st.session_state.messages.append(
                ModelRequest(parts=[UserPromptPart(content=user_input)])
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                await run_agent_with_streaming(user_input)

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        if existing_data:
            st.info("The knowledge base is ready! Start asking questions below.")
        else:
            st.info("Please process a URL first to start chatting!")



if __name__ == "__main__":
    asyncio.run(main())