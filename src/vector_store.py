"""
Medical QnA RAG System - Vector Store Module
"""
from typing import List
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import Config


class VectorStore:
    """Manages vector database operations"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.vectorstore = None
    
    def create_vectorstore(self, documents: List[Document]):
        """
        Create a new vector store from documents
        
        Args:
            documents: List of Document objects to index
        """
        if not documents:
            print("No documents to index")
            return
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=Config.COLLECTION_NAME,
            persist_directory=Config.VECTOR_DB_PATH
        )
        print(f"Created vector store with {len(documents)} document chunks")
    
    def load_vectorstore(self):
        """Load existing vector store from disk"""
        self.vectorstore = Chroma(
            collection_name=Config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=Config.VECTOR_DB_PATH
        )
        print("Loaded existing vector store")
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar Document objects
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized")
        
        k = k or Config.TOP_K_RESULTS
        return self.vectorstore.similarity_search(query, k=k)
