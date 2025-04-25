import tiktoken
from transformers import AutoTokenizer

from config import ModelConfig, ModelType

# Tokenizer for local Mistral-based models
_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

# Tokenizer used for OpenAI models (e.g., GPT-3.5/4/4o)
_openai_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(model: ModelConfig, messages: list[dict]) -> int:
    """
    Estimate the total number of tokens in a list of chat messages.

    Args:
        model: The model configuration, used to select the appropriate tokenizer.
        messages: List of chat messages (each must have a "content" field).

    Returns:
        An integer representing the total number of tokens in all message contents.

    Notes:
        - For OpenAI models, uses the cl100k_base tokenizer (used by GPT-3.5, GPT-4, GPT-4o).
        - For all other models (e.g. Mistral), uses HuggingFace's AutoTokenizer.
        - Does not account for role/metadata token overhead (e.g., OpenAI's per-message costs).
    """
    full_text = "".join([m["content"] for m in messages])

    if model.model_type == ModelType.OPENAI:
        return len(_openai_encoding.encode(full_text))
    else:
        return len(_tokenizer.encode(full_text))
