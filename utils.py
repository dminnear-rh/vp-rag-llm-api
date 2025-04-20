from transformers import AutoTokenizer

_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")


def count_tokens(messages: list[dict]) -> int:
    full_text = "".join([m["content"] for m in messages])
    return len(_tokenizer.encode(full_text))
