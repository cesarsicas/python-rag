# RAG Assistant

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your own text files. It uses OpenAI embeddings and FAISS for semantic search, and GPT-4o-mini to generate answers grounded in your knowledge base.

## How it works

1. Loads all `.txt` files from the `rag_files/` folder
2. Splits them into chunks and generates embeddings
3. Stores the embeddings in a local FAISS vector store
4. On each user query, retrieves the most relevant chunks and sends them as context to the LLM

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root folder with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

5. Add your `.txt` files to the `rag_files/` folder.

## Usage

```bash
python script.py
```

Type your question and the assistant will answer using only the content from your files. Type `exit` to quit.
