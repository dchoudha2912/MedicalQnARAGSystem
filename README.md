# Medical QnA RAG System

A Retrieval-Augmented Generation (RAG) based Medical Question & Answer bot that provides accurate medical information by combining document retrieval with large language models.

## Overview

This system uses RAG technology to answer medical questions by:
1. Retrieving relevant medical information from a vector database
2. Using the retrieved context to generate accurate, contextual answers via OpenAI's GPT models
3. Providing source references for transparency

## Features

- 🔍 **Semantic Search**: Uses embeddings to find relevant medical information
- 🤖 **AI-Powered Answers**: Leverages OpenAI's GPT for natural language responses
- 📚 **Source Attribution**: Shows which documents were used to generate answers
- 💾 **Persistent Storage**: Vector database persists across sessions
- 🎯 **Easy to Extend**: Simply add more medical documents to improve coverage

## Architecture

```
User Question → Vector Store (Retrieval) → LLM (Generation) → Answer
                    ↓
              Medical Documents
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/dchoudha2912/MedicalQnARAGSystem.git
   cd MedicalQnARAGSystem
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Interactive Mode

Run the system in interactive mode to ask questions:

```bash
python main.py
```

The system will:
1. Load or create the vector database
2. Start an interactive session
3. Allow you to ask medical questions

Example interaction:
```
Your question: What are the symptoms of diabetes?

Searching for relevant information...

Answer: The symptoms of diabetes include increased thirst and frequent 
urination, extreme hunger, unexplained weight loss, fatigue, blurred 
vision, slow-healing sores, and frequent infections...

Sources: diabetes.txt
```

### Setup Mode

To rebuild the vector database:

```bash
python main.py setup
```

Or specify a custom data directory:

```bash
python main.py setup /path/to/medical/documents
```

## Adding Medical Documents

1. Add `.txt` files to the `data/` directory
2. Run the setup command to reindex:
   ```bash
   python main.py setup
   ```

The system currently includes example documents on:
- Diabetes Mellitus
- Hypertension
- Common Cold and Influenza

## Project Structure

```
MedicalQnARAGSystem/
├── main.py                 # Main application entry point
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings
│   ├── document_loader.py # Document loading and chunking
│   ├── vector_store.py    # Vector database operations
│   └── rag_pipeline.py    # RAG pipeline implementation
├── data/                  # Medical documents directory
│   ├── diabetes.txt
│   ├── hypertension.txt
│   └── cold_flu.txt
├── requirements.txt       # Python dependencies
├── .env.example          # Example environment variables
└── README.md             # This file
```

## Technology Stack

- **LangChain**: Framework for LLM applications
- **OpenAI**: Embeddings and language model
- **ChromaDB**: Vector database for document storage
- **Python-dotenv**: Environment variable management

## Configuration

Edit `src/config.py` or set environment variables to customize:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `EMBEDDING_MODEL`: Embedding model (default: text-embedding-ada-002)
- `LLM_MODEL`: Language model (default: gpt-3.5-turbo)
- `CHUNK_SIZE`: Document chunk size (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `TOP_K_RESULTS`: Number of documents to retrieve (default: 3)

## Important Notes

⚠️ **Medical Disclaimer**: This system is for educational and informational purposes only. Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.

## Future Enhancements

- [ ] Web interface using Streamlit or Flask
- [ ] Support for PDF and other document formats
- [ ] Multi-language support
- [ ] Conversation history and context
- [ ] Fine-tuned medical language models
- [ ] Enhanced source citation with page numbers

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.