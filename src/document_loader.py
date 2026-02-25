"""
Medical QnA RAG System - Document Loader Module
"""
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config import Config


class DocumentLoader:
    """Handles loading and processing of medical documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load all text files from a directory
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of Document objects
        """
        documents = []
        
        if not os.path.exists(directory_path):
            print(f"Directory {directory_path} does not exist")
            return documents
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": filename}
                        ))
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of split Document objects
        """
        return self.text_splitter.split_documents(documents)
