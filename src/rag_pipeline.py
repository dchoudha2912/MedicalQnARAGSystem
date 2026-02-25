"""
Medical QnA RAG System - RAG Pipeline Module
"""
from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from src.config import Config
from src.vector_store import VectorStore


class RAGPipeline:
    """Main RAG pipeline for question answering"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            openai_api_key=Config.OPENAI_API_KEY,
            temperature=0.7
        )
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful medical assistant. Answer the user's question based on the provided context.
If the context doesn't contain enough information to answer the question, say so clearly.
Always provide accurate medical information and remind users to consult healthcare professionals for personalized advice.

Context:
{context}"""),
            ("human", "{question}")
        ])
    
    def answer_question(self, question: str) -> dict:
        """
        Answer a question using RAG
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and source documents
        """
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate answer using LLM
        messages = self.prompt_template.format_messages(
            context=context,
            question=question
        )
        
        response = self.llm.invoke(messages)
        
        return {
            "answer": response.content,
            "sources": [doc.metadata.get("source", "Unknown") for doc in relevant_docs]
        }
