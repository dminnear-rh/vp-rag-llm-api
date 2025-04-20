import json
import logging

import httpx
import openai

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


async def get_completion(
    config: AppConfig, model_name: str, messages: list[dict]
) -> str:
    model = config.models[model_name]

    logger.debug(
        f"📝 Prompt sent to model '{model_name}':\n{json.dumps(messages, indent=2)}"
    )
    token_count = count_tokens(messages)
    logger.info(f"📏 Tokens in request: {token_count}")

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
            result = resp.json()
            answer = result["choices"][0]["message"]["content"]
            logger.debug(f"🧠 Response from vLLM:\n{answer}")
            logger.info(
                f"📏 Tokens in response: {count_tokens([{'role': 'assistant', 'content': answer}])}"
            )
            return answer

    elif model.model_type == "openai":
        openai.api_key = config.openai_api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.4,
            top_p=0.9,
        )
        result = response.choices[0].message["content"]
        logger.debug(f"🧠 Response from OpenAI:\n{result}")
        logger.info(
            f"📏 Tokens in response: {count_tokens([{'role': 'assistant', 'content': result}])}"
        )
        return result

    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")


async def stream_completion(config: AppConfig, model_name: str, messages: list[dict]):
    model = config.models[model_name]

    logger.debug(
        f"📝 [stream] Prompt sent to model '{model_name}':\n{json.dumps(messages, indent=2)}"
    )
    token_count = count_tokens(messages)
    logger.info(f"📏 [stream] Tokens in request: {token_count}")

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
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n"

    elif model.model_type == "openai":
        openai.api_key = config.openai_api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=3072,
            temperature=0.5,
            top_p=0.9,
            stream=True,
        )

        full_response = ""
        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                full_response += delta["content"]
                yield f"data: {json.dumps({'content': delta['content']})}\n"

        logger.debug(f"🧠 [stream] Full response:\n{full_response}")
        logger.info(
            f"📏 [stream] Tokens in response: {count_tokens([{'role': 'assistant', 'content': full_response}])}"
        )
        yield "data: [DONE]\n"

    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")
