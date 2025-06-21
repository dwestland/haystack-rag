# SMS RAG Pipeline

A Retrieval-Augmented Generation (RAG) system for querying SMS conversations using Haystack and ChromaDB.

## Setup

### Start with SMS messages exported from Messive in a file named "missive-conversations.zip"

1. **Create virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Place your SMS data** in the `2025-05` directory as JSON files

2. **Run the RAG pipeline:**

   ```bash
   python rag_pipeline.py
   ```

3. **Ask questions** about your SMS conversations:
   - "What did Dylan say about FastStager?"
   - "Show me conversations with phone number +1 (626) 325-0897"
   - "What were the recent messages about virtual staging?"

## Features

- ✅ Processes all JSON files recursively
- ✅ Groups messages by conversation ID
- ✅ Preserves conversation threading
- ✅ Includes participant metadata
- ✅ Persistent ChromaDB storage
- ✅ Interactive command-line interface
- ✅ Uses latest Haystack 2.x patterns

## Architecture

- **Data Processing**: Loads JSON files and groups messages by conversation
- **Indexing Pipeline**: OpenAI embeddings → ChromaDB storage
- **Query Pipeline**: Text embedder → ChromaEmbeddingRetriever → ChatPromptBuilder → GPT-4 Turbo
