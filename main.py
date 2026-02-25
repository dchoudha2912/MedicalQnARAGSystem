"""
Medical QnA RAG System - Main Application
"""
import os
import sys
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.config import Config


def setup_vectorstore(data_dir: str = "./data"):
    """
    Setup vector store by loading and indexing documents
    
    Args:
        data_dir: Directory containing medical documents
    """
    print("Setting up vector store...")
    
    # Load documents
    loader = DocumentLoader()
    documents = loader.load_from_directory(data_dir)
    
    if not documents:
        print(f"No documents found in {data_dir}")
        return None
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents
    split_docs = loader.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")
    
    # Create vector store
    vector_store = VectorStore()
    vector_store.create_vectorstore(split_docs)
    
    return vector_store


def load_existing_vectorstore():
    """Load existing vector store from disk"""
    vector_store = VectorStore()
    vector_store.load_vectorstore()
    return vector_store


def run_interactive_mode():
    """Run interactive Q&A session"""
    print("\n" + "="*60)
    print("Medical QnA RAG System")
    print("="*60)
    
    # Check if API key is set
    if not Config.OPENAI_API_KEY:
        print("\nError: OPENAI_API_KEY not set!")
        print("Please set your OpenAI API key in .env file")
        return
    
    # Check if vector store exists
    vectorstore_exists = os.path.exists(Config.VECTOR_DB_PATH)
    
    if not vectorstore_exists:
        print("\nVector store not found. Creating new one...")
        vector_store = setup_vectorstore()
        if vector_store is None:
            print("\nFailed to setup vector store. Please add documents to ./data directory")
            return
    else:
        print("\nLoading existing vector store...")
        vector_store = load_existing_vectorstore()
    
    # Initialize RAG pipeline
    rag = RAGPipeline(vector_store)
    
    print("\nSystem ready! Ask your medical questions (type 'quit' or 'exit' to stop)")
    print("-"*60)
    
    while True:
        try:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching for relevant information...")
            result = rag.answer_question(question)
            
            print(f"\nAnswer: {result['answer']}")
            
            if result['sources']:
                print(f"\nSources: {', '.join(result['sources'])}")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            # Setup vector store
            data_dir = sys.argv[2] if len(sys.argv) > 2 else "./data"
            setup_vectorstore(data_dir)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py [setup]")
    else:
        # Run interactive mode
        run_interactive_mode()


if __name__ == "__main__":
    main()
