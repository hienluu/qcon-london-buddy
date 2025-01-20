<!-- @format -->
This is a simple RAG based GenAI application called QCon London Buddy. This simple applications aims to help attendees to discover and ask questions and details about the conference, such as track, sessions, speakers, etc.  Attendees can us NLP to ask for recommendations based on their interest and background.

It crawls contents from https://qconlondon.com, stores them in Chroma DB, uses an embedding model to generate embedding, and uses an LLM to help with answering the questions.  The interaction is through simple chat interface.  This project leverages [PydanticAI](https://ai.pydantic.dev/agents/) as an AI agents framework.

> **Note:** Before running the project, make sure to install the dependencies listed in `requirements.txt`. Focus on the `gen-rag-crawl` module as it is the final dynamic RAG system to look into.


### Resources:
This project was based on these resources:
* https://github.com/pdichone/crawl4ai-rag-system.git
* https://github.com/coleam00/ottomator-agents.git
* [The Future of RAG is Agentic](https://www.youtube.com/watch?v=_R-ff4ZMLC8)
* [Turn ANY Website into AI Knowledge in seconds with Cral4AI](https://www.youtube.com/watch?v=RNlo21BQ68E)
* [Chroma](https://docs.trychroma.com/docs/overview/introduction)
* [Chroma UI #1](https://github.com/thakkaryash94/chroma-ui)
* [Chroma UI #2](https://github.com/thakkaryash94/chroma-ui)
* [Streamlit](https://docs.streamlit.io/)
* [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)
* [PydanticAI](https://ai.pydantic.dev/agents/)
* [OpenAI Pricing](https://openai.com/api/pricing/)
* [Crawl4AI](https://docs.crawl4ai.com/core/page-interaction/)