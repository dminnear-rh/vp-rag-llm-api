import tiktoken
from transformers import AutoTokenizer

from config import ModelConfig, ModelType

_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
_openai_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(model: ModelConfig, messages: list[dict]) -> int:
    full_text = "".join([m["content"] for m in messages])
    if model.model_type == ModelType.OPENAI:
        return len(_openai_encoding.encode(full_text))
    else:
        return len(_tokenizer.encode(full_text))
