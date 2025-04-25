import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from llm import get_completion, stream_completion
from models import RAGRequest
from retrieval import build_messages

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/models")
async def list_models(request: Request):
    """
    List all available LLM models registered in the app configuration.

    Returns:
        A JSON object containing:
            - default_model: The name of the default model
            - models: A list of model configs, each including:
                - name: Model identifier
                - model_type: Type of model (e.g., "openai", "vllm")
                - url: Model endpoint (if applicable)
                - max_total_tokens: Token limit for request + response
    """
    config = request.app.state.config
    return {
        "default_model": config.default_model,
        "models": [
            {
                "name": m.name,
                "model_type": m.model_type,
                "url": m.url,
                "max_total_tokens": m.max_total_tokens,
            }
            for m in config.models.values()
        ],
    }


@router.post("/rag-query")
async def rag_query(request: Request, payload: RAGRequest):
    """
    Handle a non-streaming Retrieval-Augmented Generation (RAG) query.

    Combines search results from Qdrant and conversation history to build a prompt
    and sends it to the selected LLM to generate a full answer.

    Args:
        request: FastAPI request context, used to access app config.
        payload: RAGRequest containing the user question, optional model override, and history.

    Returns:
        A JSON response with the model's generated answer:
            {"answer": "<text>"}
    """
    config = request.app.state.config
    try:
        model = config.models[payload.model]
    except:
        model = config.models[config.default_model]

    messages = build_messages(config, model, payload.question, payload.history)
    answer = await get_completion(config, model, messages)
    return {"answer": answer}


@router.post("/rag-query/stream")
async def rag_query_stream(request: Request, payload: RAGRequest):
    """
    Handle a streaming Retrieval-Augmented Generation (RAG) query.

    Builds a prompt using vector search results and chat history,
    then streams token-by-token completion results as Server-Sent Events (SSE).

    Args:
        request: FastAPI request context, used to access app config.
        payload: RAGRequest containing the user question, optional model override, and history.

    Returns:
        StreamingResponse: A text/event-stream response suitable for real-time UIs.
    """
    config = request.app.state.config
    try:
        model = config.models[payload.model]
    except:
        model = config.models[config.default_model]

    messages = build_messages(config, model, payload.question, payload.history)

    return StreamingResponse(
        stream_completion(config, model, messages), media_type="text/event-stream"
    )
