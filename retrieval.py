import logging
from typing import List

from qdrant_client import QdrantClient

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Vector-store retrieval + re-ranking
# ---------------------------------------------------------------------
def get_context(
    config: AppConfig,
    query: str,
    limit: int = 60,
    top_n: int = 10,
) -> List[str]:
    """
    1. Embed query
    2. Similarity search in Qdrant
    3. Cross-encoder re-rank
    4. Return top-N passage strings
    """
    logger.debug(
        f"🔍 get_context(collection={config.qdrant_collection!r}, "
        f"limit={limit}, top_n={top_n}) - query={query!r}"
    )

    # 1️⃣ Embed query
    embedded_query = config.embedder.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()
    logger.debug(
        f"🧩 Embedded query - dim={len(embedded_query)}, "
        f"first5={embedded_query[:5]}"
    )

    # 2️⃣ Similarity search
    qdrant = QdrantClient(url=config.qdrant_url)
    search_result = qdrant.search(
        collection_name=config.qdrant_collection,
        query_vector=embedded_query,
        limit=limit,
    )
    logger.debug(f"🗂  Qdrant returned {len(search_result)} hits")

    if not search_result:
        logger.warning("⚠️  Qdrant search returned no hits at all")
        return []

    # Extract text from payload.  Fallback if the key isn't literally "text".
    docs: list[str] = []
    for i, hit in enumerate(search_result):
        payload = hit.payload or {}
        # Most ingest pipelines store passage text under one of these keys
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or " ".join(str(v) for v in payload.values() if isinstance(v, str))
        )
        docs.append(text)
        logger.debug(f"    • Hit #{i:02d}  score={hit.score:.4f}  len={len(text)}")

    # 3️⃣ Cross-encoder re-rank (optional)
    if hasattr(config, "cross_encoder") and config.cross_encoder is not None:
        pairs = [(query, doc) for doc in docs]
        scores = config.cross_encoder.predict(pairs)
        reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logger.debug(
            "🤖 Cross-encoder scores (top-5): "
            + ", ".join(f"{s:.3f}" for s, _ in reranked[:5])
        )
        docs = [doc for _, doc in reranked]

    # 4️⃣ Return top-N
    top_docs = docs[:top_n]
    logger.debug(
        "📚 Returning %d docs. Combined context length = %d chars",
        len(top_docs),
        sum(len(d) for d in top_docs),
    )
    return top_docs


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------
def build_messages(
    question: str, retrieved_docs: list[str], history: list[str]
) -> list[dict]:
    context = "\n---\n".join(retrieved_docs)

    system_message = (
        "You are an expert in Red Hat OpenShift and Kubernetes application architecture, "
        "specializing in the design, implementation, and validation of OpenShift Validated Patterns.\n\n"
        "The user is seeking to better understand OpenShift Validated Patterns — including how they are built, "
        "tested, used across industries, or extended for new use cases. Use the following documentation and source "
        "content to generate a clear and practical answer.\n\n"
        "Your response should:\n"
        "- Reference relevant concepts or implementation details from the provided context\n"
        "- Be technically accurate and aimed at architects, engineers, or contributors\n"
        "- Include examples or explanations using OpenShift-native tools and practices (e.g., GitOps, Operators, Pipelines, Secrets management)"
    )

    base = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Context:\n{context}"},
    ]

    return trim_history_for_tokens(base, history, question, max_tokens=25_000)


# ---------------------------------------------------------------------
# History-trimming helper
# ---------------------------------------------------------------------
def trim_history_for_tokens(
    messages: list[dict], history: list[str], question: str, max_tokens: int
) -> list[dict]:
    messages = messages[:]
    for turn in history:
        messages.append({"role": "user", "content": turn})
        if (
            count_tokens(
                messages + [{"role": "user", "content": f"Question: {question}"}]
            )
            > max_tokens
        ):
            # Remove the oldest user-supplied context (index 2) to stay under the cap
            messages.pop(2)
            break
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages
