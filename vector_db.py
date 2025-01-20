# vector_db.py
import os
import chromadb
from chromadb.config import Settings, System
from dotenv import load_dotenv

load_dotenv()

VECTOR_DB_LOCATION=os.getenv('VECTOR_DB_LOCATION')
VECTOR_DB_COLLECTION_NAME=os.getenv('VECTOR_DB_COLLECTION_NAME')

def get_chroma_client():
    return chromadb.PersistentClient(
        path=VECTOR_DB_LOCATION,
        settings=Settings(
            allow_reset=True, anonymized_telemetry=False, is_persistent=True
        ),
    )


def init_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=VECTOR_DB_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,  # We're providing our own embeddings
    )
