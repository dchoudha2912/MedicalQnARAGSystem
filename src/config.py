"""
Medical QnA RAG System - Configuration Module
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Configuration class for the RAG system"""
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    VECTOR_DB_PATH = "./chroma_db"
    COLLECTION_NAME = "medical_documents"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 3
