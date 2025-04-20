from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from llm import get_completion, stream_completion
from retrieval import build_messages, get_context
from utils import count_tokens

from models import RAGRequest

router = APIRouter()


@router.post("/rag-query")
async def rag_query(request: Request, payload: RAGRequest):
    config = request.app.state.config
    model_name = payload.model or config.default_model
    context_chunks = get_context(config, payload.question)
    messages = build_messages(payload.question, context_chunks, payload.history)

    prompt_tokens = count_tokens(messages)
    print(f"Prompt tokens: {prompt_tokens}")

    answer = await get_completion(config, model_name, messages)
    return {"answer": answer}


@router.post("/rag-query/stream")
async def rag_query_stream(request: Request, payload: RAGRequest):
    config = request.app.state.config
    model_name = payload.model or config.default_model
    context_chunks = get_context(config, payload.question)
    messages = build_messages(payload.question, context_chunks, payload.history)

    prompt_tokens = count_tokens(messages)
    print(f"[stream] Prompt tokens: {prompt_tokens}")

    return StreamingResponse(
        stream_completion(config, model_name, messages), media_type="text/event-stream"
    )
