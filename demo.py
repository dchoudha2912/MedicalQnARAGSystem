"""
Demo script showing the RAG system workflow
This demonstrates the system's capabilities without requiring an API key
"""

print("="*70)
print("Medical QnA RAG System - Demo Workflow")
print("="*70)

print("\n1. Document Loading Phase")
print("-" * 70)

from src.document_loader import DocumentLoader

loader = DocumentLoader()
documents = loader.load_from_directory("./data")

print(f"✓ Loaded {len(documents)} medical documents:")
for doc in documents:
    source = doc.metadata.get('source', 'Unknown')
    length = len(doc.page_content)
    print(f"  - {source}: {length} characters")

print("\n2. Document Chunking Phase")
print("-" * 70)

split_docs = loader.split_documents(documents)
print(f"✓ Split documents into {len(split_docs)} chunks for better retrieval")
print(f"  Average chunk size: {sum(len(d.page_content) for d in split_docs) // len(split_docs)} characters")

print("\n3. Sample Chunk Preview")
print("-" * 70)

sample_chunk = split_docs[0]
print(f"Source: {sample_chunk.metadata.get('source', 'Unknown')}")
print(f"Content preview (first 200 chars):\n")
print(sample_chunk.page_content[:200] + "...\n")

print("4. System Architecture")
print("-" * 70)
print("""
The RAG system follows this workflow:

User Question
     ↓
[1. Embedding] - Convert question to vector
     ↓
[2. Retrieval] - Find similar document chunks in vector DB
     ↓
[3. Context Assembly] - Combine relevant chunks
     ↓
[4. Generation] - LLM generates answer using context
     ↓
Answer + Sources
""")

print("5. Example Queries the System Can Answer")
print("-" * 70)
example_queries = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What's the difference between common cold and flu?",
    "What medications are used to treat high blood pressure?",
    "How can Type 2 diabetes be prevented?",
    "What are the complications of untreated diabetes?",
]

for i, query in enumerate(example_queries, 1):
    print(f"{i}. {query}")

print("\n6. How to Use the System")
print("-" * 70)
print("""
Step 1: Set up your OpenAI API key in .env file
        cp .env.example .env
        # Edit .env and add your OPENAI_API_KEY

Step 2: Run the interactive system
        python main.py

Step 3: Ask your medical questions!
        Your question: What are the symptoms of diabetes?

The system will:
  - Search the vector database for relevant information
  - Retrieve the top 3 most relevant document chunks
  - Use GPT to generate a comprehensive answer
  - Show you the source documents used
""")

print("7. Adding More Medical Knowledge")
print("-" * 70)
print("""
To expand the system's knowledge:

1. Add new .txt files to the data/ directory
2. Run: python main.py setup
3. The system will reindex all documents

Supported topics can include:
- Disease information and symptoms
- Treatment options and medications
- Prevention strategies
- Diagnostic procedures
- Medical terminology explanations
""")

print("\n" + "="*70)
print("Demo Complete!")
print("="*70)
print("\nThe RAG system is ready to answer medical questions.")
print("To run it with real queries, set up your OpenAI API key and run:")
print("  python main.py")
