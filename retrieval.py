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
    top_n: int = 12,
) -> List[str]:
    """Return top passages plus immediate neighbours for richer context."""

    logger.debug(
        "🔍 get_context(collection=%r, limit=%d, top_n=%d) – query=%r",
        config.qdrant_collection,
        limit,
        top_n,
        query,
    )

    # 1️⃣ Embed query
    embedded_query = config.embedder.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()
    logger.debug(
        "🧩 Embedded query dim=%d first5=%s", len(embedded_query), embedded_query[:5]
    )

    # 2️⃣ Similarity search
    qdrant = QdrantClient(url=config.qdrant_url)
    hits = qdrant.search(
        collection_name=config.qdrant_collection,
        query_vector=embedded_query,
        limit=limit,
    )
    logger.debug("🗂  Qdrant returned %d hits", len(hits))
    if not hits:
        logger.warning("⚠️  No vector hits returned from Qdrant")
        return []

    # Map idx → text
    idx_to_text: dict[int, str] = {}
    for idx, hit in enumerate(hits):
        payload = hit.payload or {}
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or " ".join(v for v in payload.values() if isinstance(v, str))
        )
        idx_to_text[idx] = text
        logger.debug("    • Hit #%02d  score=%.4f  len=%d", idx, hit.score, len(text))

    # 3️⃣ Cross-encoder re-rank (if configured)
    scored_pairs = [(1.0, txt, i) for i, txt in idx_to_text.items()]
    if getattr(config, "cross_encoder", None):
        ce_scores = config.cross_encoder.predict(
            [(query, t) for t in idx_to_text.values()]
        )
        scored_pairs = [
            (s, t, i)
            for (s, t), i in zip(
                zip(ce_scores, idx_to_text.values()), idx_to_text.keys()
            )
        ]
        logger.debug(
            "🤖 Cross-encoder top-5: %s",
            ", ".join(f"{s:.3f}" for s, _, _ in sorted(scored_pairs, reverse=True)[:5]),
        )

    # 4️⃣ Select top-N indices and include neighbours (idx±1)
    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    primary_idxs = [i for _, _, i in scored_pairs[:top_n]]
    neighbour_idxs = {i - 1 for i in primary_idxs if i - 1 >= 0} | {
        i + 1 for i in primary_idxs if i + 1 < len(hits)
    }
    final_idxs = sorted(set(primary_idxs) | neighbour_idxs)

    docs = [idx_to_text[i] for i in final_idxs]
    logger.debug(
        "📚 Returning %d docs (with neighbours). Combined chars=%d",
        len(docs),
        sum(len(d) for d in docs),
    )

    # Cap the number of chunks to prevent oversized prompts
    return docs[: top_n * 3]


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
