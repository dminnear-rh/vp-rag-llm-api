import json
import httpx
import openai
from config import AppConfig


async def get_completion(
    config: AppConfig, model_name: str, messages: list[dict]
) -> str:
    model = config.models[model_name]

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
                headers=(
                    {"Authorization": f"Bearer {model.api_key}"}
                    if model.api_key
                    else None
                ),
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    elif model.model_type == "openai":
        openai.api_key = model.api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=2048,
            temperature=0.4,
            top_p=0.9,
        )
        return response.choices[0].message["content"]

    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")


async def stream_completion(config: AppConfig, model_name: str, messages: list[dict]):
    model = config.models[model_name]

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
                headers=(
                    {"Authorization": f"Bearer {model.api_key}"}
                    if model.api_key
                    else None
                ),
            ) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        yield f"{line}\n"

    elif model.model_type == "openai":
        openai.api_key = model.api_key

        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            max_tokens=3072,
            temperature=0.5,
            top_p=0.9,
            stream=True,
        )

        for chunk in response:
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                yield f"data: {json.dumps({'content': delta['content']})}\n"

        yield "data: [DONE]\n"

    else:
        raise ValueError(f"Unsupported model type: {model.model_type}")
