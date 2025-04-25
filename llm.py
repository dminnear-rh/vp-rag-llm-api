import json
import logging
from typing import AsyncGenerator

import httpx
from openai import OpenAI

from config import AppConfig, ModelConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------
def _log_request(model: ModelConfig, messages: list[dict], stream: bool = False):
    tag = "[stream] " if stream else ""
    logger.debug(
        f"📝 {tag}Prompt sent to model '{model.name}':\n{json.dumps(messages, indent=2)}"
    )
    logger.info(f"📏 {tag}Tokens in request: {count_tokens(model, messages)}")


def _log_response(model: ModelConfig, answer: str, stream: bool = False):
    tag = "[stream] " if stream else ""
    logger.debug(f"🧠 {tag}Response from {model.name}:\n{answer}")
    logger.info(
        f"📏 {tag}Tokens in response: "
        f"{count_tokens(model, [{'role': 'assistant', 'content': answer}])}"
    )


# ---------------------------------------------------------------------
# Non‑streaming
# ---------------------------------------------------------------------
async def get_completion(
    config: AppConfig, model: ModelConfig, messages: list[dict]
) -> str:
    _log_request(model, messages)

    # -------- vLLM ----------------------------------------------------
    if model.model_type == "vllm":
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{model.url}/v1/chat/completions",
                json={
                    "model": model.name,
                    "messages": messages,
                    "max_tokens": config.max_response_tokens,
                    "temperature": 0.4,
                    "top_p": 0.9,
                },
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]
            _log_response(model, result)
            return result

    # -------- OpenAI --------------------------------------------------
    if model.model_type == "openai":
        client = OpenAI(api_key=config.openai_api_key)

        response = client.chat.completions.create(
            model=model.name,
            messages=messages,
            max_completion_tokens=config.max_response_tokens,
            temperature=0.4,
            top_p=0.9,
        )
        result = response.choices[0].message.content
        _log_response(model, result)
        return result

    raise ValueError(f"Unsupported model type: {model.model_type}")


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------
async def stream_completion(
    config: AppConfig, model: ModelConfig, messages: list[dict]
) -> AsyncGenerator[str, None]:
    _log_request(model, messages, stream=True)

    # -------- vLLM ----------------------------------------------------
    if model.model_type == "vllm":
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{model.url}/v1/chat/completions",
                json={
                    "model": model.name,
                    "messages": messages,
                    "max_tokens": config.max_response_tokens,
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "stream": True,
                },
            ) as r:
                full_resp = ""
                async for raw in r.aiter_lines():
                    if not raw or not raw.startswith("data: "):
                        continue

                    payload = raw[6:]  # strip "data: "
                    if payload.strip() == "[DONE]":
                        yield "data: [DONE]\n"
                        break

                    try:
                        obj = json.loads(payload)
                        delta = obj["choices"][0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                            full_resp += token
                            yield f"data: {json.dumps({'content': token})}\n"
                    except json.JSONDecodeError:
                        continue  # ignore malformed keep‑alive lines

                _log_response(model, full_resp, stream=True)
        return

    # -------- OpenAI --------------------------------------------------
    if model.model_type == "openai":
        client = OpenAI(api_key=config.openai_api_key)
        full_resp = ""

        # new‑style streaming generator
        for chunk in client.chat.completions.create(
            model=model.name,
            messages=messages,
            max_completion_tokens=config.max_response_tokens,
            temperature=0.5,
            top_p=0.9,
            stream=True,
        ):
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token = delta.content
                full_resp += token
                yield f"data: {json.dumps({'content': token})}\n"

        yield "data: [DONE]\n"
        _log_response(model, full_resp, stream=True)
        return

    raise ValueError(f"Unsupported model type: {model.model_type}")
