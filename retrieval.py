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
    # Vector search
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

    # Cross-encoder re-rank
    scores = [p.score for p in search]
    if getattr(config, "cross_encoder", None):
        scores = config.cross_encoder.predict(
            [(query, p.payload["page_content"]) for p in search]
        )

    ranked_idx = sorted(range(len(search)), key=lambda i: scores[i], reverse=True)

    # Pick primary_k top chunks
    top_k = [search[ranked_idx[i]] for i in range(min(primary_k, len(search)))]

    # Get sources for top_k vectors
    top_k_sources: Set[str] = set()
    for p in top_k:
        source = p.payload["metadata"]["source"]
        top_k_sources.add(source)

    # Retrieve full source document for top_k_sources
    top_k_contents: List[str] = []
    for source in top_k_sources:
        points, _ = qclient.scroll(
            collection_name=config.qdrant_collection,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": source}},
                    {
                        "key": "metadata.chunk_id",
                        "range": {
                            "gte": 0,
                            "lte": 100,
                        },
                    },
                ]
            },
        )

        # Sort by chunk_id
        sorted_points = sorted(points, key=lambda p: p.payload["metadata"]["chunk_id"])
        content = f"Source: {source}\n"
        for point in sorted_points:
            content += f"\n{point.payload['page_content']}"
        top_k_contents.append(content)

    # Remove least relevant source docs if we overshoot max_tokens
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
    base = [
        {"role": "system", "content": config.system_message},
    ]
    user_question = {"role": "user", "content": f"Question: {question}"}

    # Figure out how many tokens we can devote to context and history
    base_tokens = count_tokens(model, base)
    user_question_tokens = count_tokens(model, [user_question])
    max_tokens = (
        model.max_total_tokens
        - base_tokens
        - user_question_tokens
        - config.max_response_tokens
        - 100
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
    # Alternate roles: user → assistant → user → assistant → ...
    roles = ["user", "assistant"]
    trimmed_history = [
        {"role": roles[i % 2], "content": turn} for i, turn in enumerate(history)
    ]

    # Trim oldest messages
    while count_tokens(model, trimmed_history) > max_tokens:
        trimmed_history.pop(0)

    return trimmed_history
