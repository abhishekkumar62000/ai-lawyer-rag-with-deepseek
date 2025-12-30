# AI Lawyer RAG

RAG-powered Q&A over uploaded legal PDFs using FAISS + LangChain for retrieval and Groq for answers. Embeddings come from a lightweight local Ollama model.

## Prerequisites
- Python 3.10+
- Ollama installed and running (Windows: Ollama Desktop)
- Groq API key

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Pull an embedding model for Ollama (recommended `nomic-embed-text`):
   ```bash
   ollama pull nomic-embed-text
   ```
3. Configure environment:
   - Copy `.env.example` to `.env` and fill values, or set in shell:
   ```powershell
   $Env:GROQ_API_KEY = "<your_key>"
   $Env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
   # Optional: explicitly choose a Groq model
   $Env:GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

## Run
```powershell
streamlit run frontend.py
# or
streamlit run main.py
```

## Features
- Upload a PDF and build a fresh FAISS index per session
- Top-K slider to control retrieved chunks
- Token-aware context construction to respect model limits
- Source citations with chunk previews and page info
- Robust Groq model selection and automatic fallback on decommission

## Tips
- For faster, accurate retrieval, keep `OLLAMA_EMBED_MODEL` set to `nomic-embed-text` or `mxbai-embed-large`.
- Large PDFs: reduce `Top K` and ask specific questions to improve focus.

## Troubleshooting
- "Model not found" in Ollama: run `ollama pull <model>` and ensure Ollama is running.
- Groq decommission errors: set `GROQ_MODEL` to a supported model or rely on automatic selection.
- Memory errors with large models: use lightweight embedding models; avoid chat models for embeddings.