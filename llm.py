import json
import logging
from typing import AsyncGenerator

import httpx
import openai

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------
def _log_request(model_name: str, messages: list[dict], stream: bool = False):
    tag = "[stream] " if stream else ""
    logger.debug(
        f"📝 {tag}Prompt sent to model '{model_name}':\n{json.dumps(messages, indent=2)}"
    )
    logger.info(f"📏 {tag}Tokens in request: {count_tokens(messages)}")


def _log_response(model_name: str, answer: str, stream: bool = False):
    tag = "[stream] " if stream else ""
    logger.debug(f"🧠 {tag}Response from {model_name}:\n{answer}")
    logger.info(
        f"📏 {tag}Tokens in response: "
        f"{count_tokens([{'role': 'assistant', 'content': answer}])}"
    )


# ------------------------------------------------------------
# Non‑streaming
# ------------------------------------------------------------
async def get_completion(
    config: AppConfig, model_name: str, messages: list[dict]
) -> str:
    model = config.models[model_name]
    _log_request(model_name, messages)

    if model.model_type == "vllm":
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{model.url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.4,
                    "top_p": 0.9,
                },
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]
            _log_response("vLLM", result)
            return result

    elif model.model_type == "openai":
        openai.api_key = config.openai_api_key
        result = (
            openai.ChatCompletion.create(
                model=model_name,
                messages=messages,
                max_tokens=2048,
                temperature=0.4,
                top_p=0.9,
            )
            .choices[0]
            .message["content"]
        )
        _log_response("OpenAI", result)
        return result

    raise ValueError(f"Unsupported model type: {model.model_type}")


# ------------------------------------------------------------
# Streaming
# ------------------------------------------------------------
async def stream_completion(
    config: AppConfig, model_name: str, messages: list[dict]
) -> AsyncGenerator[str, None]:
    model = config.models[model_name]
    _log_request(model_name, messages, stream=True)

    if model.model_type == "vllm":
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{model.url}/v1/chat/completions",
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 3072,
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
                            # Normalized tiny chunk -> matches OpenAI branch
                            yield f"data: {json.dumps({'content': token})}\n"
                    except json.JSONDecodeError:
                        continue  # ignore malformed keep‑alive lines

                _log_response("vLLM", full_resp, stream=True)
        return

    if model.model_type == "openai":
        openai.api_key = config.openai_api_key
        full_resp = ""
        for chunk in openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=3072,
            temperature=0.5,
            top_p=0.9,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                token = delta["content"]
                full_resp += token
                yield f"data: {json.dumps({'content': token})}\n"

        yield "data: [DONE]\n"
        _log_response("OpenAI", full_resp, stream=True)
        return

    raise ValueError(f"Unsupported model type: {model.model_type}")
