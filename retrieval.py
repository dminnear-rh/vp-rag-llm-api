import logging
from typing import List, Set

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint

from config import AppConfig, ModelConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


def _get_context(
    config: AppConfig,
    model: ModelConfig,
    query: str,
    max_tokens: int,
    limit: int = 60,
    primary_k: int = 10,
) -> dict:
    """
    Retrieve and construct context documents from Qdrant based on a query.

    This function performs a vector similarity search, re-ranks results using
    a cross-encoder (if configured), and fetches full source documents for the top-k matches.
    It then builds a context message formatted for LLM input.

    Args:
        config: Application settings including Qdrant and encoder configuration.
        model: The model for which context is being built (used for token budget).
        query: The user question or search prompt.
        max_tokens: Maximum token budget for the context portion of the prompt.
        limit: Maximum number of vector search hits to retrieve from Qdrant.
        primary_k: Number of top-ranked results to use for fetching full sources.

    Returns:
        A dict in the OpenAI message format containing context from source documents.
        Example: {"role": "user", "content": "## Context\n<doc text>\n---\n<doc text>"}
    """
    qclient = QdrantClient(url=config.qdrant_url)
    q_vec = config.embedder.encode(f"query: {query}", normalize_embeddings=True)

    search = qclient.query_points(
        collection_name=config.qdrant_collection,
        query=q_vec.tolist(),
        limit=limit,
    ).points

    if not search:
        logger.warning("⚠️  No vector hits returned from Qdrant")
        return {"role": "user", "content": "No relevant documents found."}

    scores = [p.score for p in search]
    if getattr(config, "cross_encoder", None):
        scores = config.cross_encoder.predict(
            [(query, p.payload["page_content"]) for p in search]
        )

    ranked_idx = sorted(range(len(search)), key=lambda i: scores[i], reverse=True)
    top_k = [search[ranked_idx[i]] for i in range(min(primary_k, len(search)))]

    top_k_sources: Set[str] = {p.payload["metadata"]["source"] for p in top_k}

    top_k_contents: List[str] = []
    for source in top_k_sources:
        points, _ = qclient.scroll(
            collection_name=config.qdrant_collection,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": source}},
                    {"key": "metadata.chunk_id", "range": {"gte": 0, "lte": 100}},
                ]
            },
        )

        sorted_points = sorted(points, key=lambda p: p.payload["metadata"]["chunk_id"])

        content = f"Source: {source}\n"
        for point in sorted_points:
            content += f"\n{point.payload['page_content']}"
        top_k_contents.append(content)

    def context_dict(context_docs: List[str]) -> dict:
        context = "\n---\n".join(context_docs)
        return {"role": "user", "content": f"## Context\n{context}"}

    while count_tokens(model, [context_dict(top_k_contents)]) > max_tokens:
        top_k_contents.pop()

    return context_dict(top_k_contents)


def build_messages(
    config: AppConfig,
    model: ModelConfig,
    question: str,
    history: List[str],
) -> List[dict]:
    """
    Build the full message sequence to send to the LLM.

    This includes:
      - The system prompt
      - Trimmed user/assistant chat history
      - A context block derived from Qdrant
      - The current user question

    Args:
        config: Application configuration including system prompt, model registry, etc.
        model: Model configuration used for token budgeting.
        question: The new user query.
        history: A list of chat history turns (alternating user/assistant messages as plain strings).

    Returns:
        A list of OpenAI-style chat messages (each with 'role' and 'content').
    """
    base = [{"role": "system", "content": config.system_message}]
    user_question = {"role": "user", "content": f"Question: {question}"}

    base_tokens = count_tokens(model, base)
    user_question_tokens = count_tokens(model, [user_question])
    max_tokens = (
        model.max_total_tokens
        - base_tokens
        - user_question_tokens
        - config.max_response_tokens
        - 100  # buffer for safety
    )
    max_context_tokens = int(0.7 * max_tokens)
    max_history_tokens = max_tokens - max_context_tokens

    history = _trim_history(model, history, max_history_tokens)
    base.extend(history)
    base.append(_get_context(config, model, question, max_context_tokens))
    base.append(user_question)

    return base


def _trim_history(
    model: ModelConfig,
    history: List[str],
    max_tokens: int,
) -> List[dict]:
    """
    Trim user/assistant message history to fit within a token budget.

    Assumes the history is a flat list of alternating user/assistant messages (strings),
    and rewraps them into OpenAI message format. Removes the oldest messages first
    until the remaining history fits within the allowed token count.

    Args:
        model: Model config used for token counting.
        history: List of message contents (must alternate user/assistant).
        max_tokens: Token budget for the history segment.

    Returns:
        A list of OpenAI-style messages with roles and content.
    """
    roles = ["user", "assistant"]
    trimmed_history = [
        {"role": roles[i % 2], "content": turn} for i, turn in enumerate(history)
    ]

    while count_tokens(model, trimmed_history) > max_tokens:
        trimmed_history.pop(0)

    return trimmed_history
