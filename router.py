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
    config = request.app.state.config
    try:
        model = config.models[payload.model]
    except:
        model = config.models[config.default_model]
    messages = build_messages(config, model, payload.question, payload.history)

    return StreamingResponse(
        stream_completion(config, model, messages), media_type="text/event-stream"
    )
