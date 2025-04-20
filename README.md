# 🔍 RAG API for Validated Patterns

This project provides a FastAPI application that serves as a Retrieval-Augmented Generation (RAG) backend, optimized for answering questions about Validated Patterns using a hybrid of vector search and LLMs.

## ✨ Features

- 🔗 **Retrieval-Augmented Generation** with Qdrant vector search
- 🧠 **LLM Integration** with [vLLM](https://github.com/vllm-project/vllm) or [OpenAI Chat Completions API](https://platform.openai.com/docs/guides/gpt)
- 📚 **Semantic re-ranking** using a cross-encoder (MiniLM)
- 💬 **Conversation history support**
- 📡 **Streaming & non-streaming completions**
- ⚙️ **Dynamic multi-model support** via `.env` or environment variables
- 📦 **Lightweight container image** (based on UBI9 Python 3.12)

## 🧪 Example Usage

```bash
curl -X POST http://localhost:8080/rag-query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How are secrets managed in validated patterns?",
    "history": []
  }'
```

Streaming:

```bash
curl -N -X POST http://localhost:8080/rag-query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the travelops pattern do?"}'
```

## 🧰 Configuration

All settings are handled via a `.env` file — and any value can also be overridden via standard environment variables:

```env
# === Qdrant Vector Store ===
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION=rag-collection

# === Default Model Name ===
DEFAULT_MODEL=mistral

# === Model Registry (JSON array)
# model_type: one of "vllm" | "openai" | "anthropic" | etc.
# max_tokens is the total usable for prompt + response.
VLLM_MODELS=[
  {
    "name": "mistral",
    "url": "http://vllm-mistral:8000",
    "api_key": "",
    "model_type": "vllm",
    "max_total_tokens": 32768
  },
  {
    "name": "llama4",
    "url": "http://vllm-llama4:8000",
    "api_key": "",
    "model_type": "vllm",
    "max_total_tokens": 32768
  },
  {
    "name": "gpt-4",
    "url": "https://api.openai.com/v1/chat/completions",
    "api_key": "sk-...",
    "model_type": "openai",
    "max_total_tokens": 8192
  }
]

# === Logging ===
LOG_LEVEL=info
```

> 📝 See `config.py` for full validation of model and environment values.

## 🐳 Running Locally

Build the container:

```bash
podman build -t rag-api .
```

Run it:

```bash
podman run -p 8080:8080 --env-file .env rag-api
```

## 📁 Project Structure

```
├── main.py          # App entrypoint
├── router.py        # Routes for /rag-query and /rag-query/stream
├── config.py        # AppConfig loading from .env
├── llm.py           # LLM routing for OpenAI/vLLM
├── retrieval.py     # Embedding + Qdrant search + cross-encoder rerank
├── models.py        # Model dataclasses + enums
├── utils.py         # Token counting, trimming, helpers
└── .env             # Local development configuration
```
