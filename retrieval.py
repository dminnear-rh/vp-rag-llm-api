import logging
from typing import List

from qdrant_client import QdrantClient

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Vector-store retrieval + neighbour expansion
# ---------------------------------------------------------------------


def get_context(
    config: AppConfig,
    query: str,
    limit: int = 60,
    top_n: int = 15,
    neighbor_window: int = 2,
) -> List[str]:
    """Return a *deduplicated* list of passages plus neighbours.

    * ``limit`` - how many vectors to pull from Qdrant initially.
    * ``top_n`` - how many chunks to take after re-ranking.
    * ``neighbor_window`` - include ``±window`` surrounding chunks for
      additional context from the same document.
    """

    logger.debug(
        "🔍 get_context(collection=%r, limit=%d, top_n=%d, window=%d) - query=%r",
        config.qdrant_collection,
        limit,
        top_n,
        neighbor_window,
        query,
    )

    # 1️⃣ Embed query
    embedded_query = config.embedder.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    # 2️⃣ Similarity search
    qdrant = QdrantClient(url=config.qdrant_url)
    hits = qdrant.search(
        collection_name=config.qdrant_collection,
        query_vector=embedded_query,
        limit=limit,
    )
    if not hits:
        logger.warning("⚠️  No vector hits returned from Qdrant")
        return []

    # idx → (text, score)
    idx_to_text: dict[int, str] = {}
    idx_to_score: dict[int, float] = {}
    for idx, hit in enumerate(hits):
        payload = hit.payload or {}
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or " ".join(v for v in payload.values() if isinstance(v, str))
        )
        idx_to_text[idx] = text
        idx_to_score[idx] = hit.score

    # 3️⃣ Cross-encoder re-rank
    scores = [idx_to_score[i] for i in range(len(idx_to_text))]  # default
    if getattr(config, "cross_encoder", None):
        scores = config.cross_encoder.predict(
            [(query, idx_to_text[i]) for i in range(len(idx_to_text))]
        )
    ranked = sorted(range(len(idx_to_text)), key=lambda i: scores[i], reverse=True)

    # 4️⃣ Collect primary + neighbour indices
    primary = ranked[:top_n]
    neighbours = {
        j
        for i in primary
        for j in range(i - neighbor_window, i + neighbor_window + 1)
        if 0 <= j < len(idx_to_text)
    }
    final_order = [i for i in ranked if i in neighbours]  # preserve CE order

    # 5️⃣ Deduplicate identical passages while preserving order
    seen: set[str] = set()
    docs: list[str] = []
    for i in final_order:
        txt = idx_to_text[i].strip()
        key = txt[:100]  # quick hash key (first 100chars)
        if key not in seen:
            seen.add(key)
            docs.append(txt)

    logger.debug(
        "📚 Returning %d unique docs. Combined chars=%d",
        len(docs),
        sum(len(d) for d in docs),
    )
    return docs


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
        "The user is seeking to better understand Validated Patterns — including how they are built, "
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
            messages.pop(2)
            break
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages
