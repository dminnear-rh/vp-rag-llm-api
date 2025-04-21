import logging
from typing import Dict, List

import numpy as np
from qdrant_client import QdrantClient

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Small helpers for Max-Marginal Relevance (MMR)
# ------------------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _mmr_order(
    vecs: Dict[int, np.ndarray], scores: Dict[int, float], lambda_param: float = 0.2
) -> List[int]:
    """Return indices ordered by Max Marginal Relevance."""
    selected: List[int] = []
    candidates = set(vecs.keys())

    while candidates:
        if not selected:
            best = max(candidates, key=lambda i: scores[i])
        else:
            best = max(
                candidates,
                key=lambda i: lambda_param * scores[i]
                - (1 - lambda_param) * max(_cos(vecs[i], vecs[j]) for j in selected),
            )
        selected.append(best)
        candidates.remove(best)
    return selected


# ------------------------------------------------------------------
# Vector-store retrieval with neighbour expansion + MMR
# ------------------------------------------------------------------


def get_context(
    config: AppConfig,
    query: str,
    limit: int = 60,
    top_n: int = 15,
    neighbor_window: int = 2,
) -> List[str]:
    """Return deduplicated passages + neighbours using MMR for diversity."""

    logger.debug(
        "🔍 get_context(collection=%r, limit=%d, top_n=%d, window=%d) – query=%r",
        config.qdrant_collection,
        limit,
        top_n,
        neighbor_window,
        query,
    )

    # 1️⃣ Embed query
    embedded_query = config.embedder.encode(
        f"query: {query}", normalize_embeddings=True
    )

    # 2️⃣ Similarity search
    qdrant = QdrantClient(url=config.qdrant_url)
    hits = qdrant.search(
        collection_name=config.qdrant_collection,
        query_vector=embedded_query.tolist(),
        limit=limit,
    )
    if not hits:
        logger.warning("⚠️  No vector hits returned from Qdrant")
        return []

    # Build maps idx → {text, score, vec}
    idx_to_text: Dict[int, str] = {}
    idx_to_vec: Dict[int, np.ndarray] = {}
    idx_to_score: Dict[int, float] = {}

    for idx, hit in enumerate(hits):
        payload = hit.payload or {}
        text = (
            payload.get("text")
            or payload.get("content")
            or payload.get("chunk")
            or " ".join(v for v in payload.values() if isinstance(v, str))
        )
        idx_to_text[idx] = text
        idx_to_vec[idx] = np.array(hit.vector, dtype=np.float32)
        idx_to_score[idx] = hit.score

    # 3️⃣ Cross-encoder re-rank (optional)
    if getattr(config, "cross_encoder", None):
        ce_scores = config.cross_encoder.predict(
            [(query, idx_to_text[i]) for i in range(len(idx_to_text))]
        )
        for i, s in enumerate(ce_scores):
            idx_to_score[i] = float(s)
        logger.debug(
            "🤖 Cross-encoder top-5: %s",
            ", ".join(
                f"{idx_to_score[i]:.3f}"
                for i in sorted(idx_to_score, key=idx_to_score.get, reverse=True)[:5]
            ),
        )

    # 4️⃣ MMR ordering for diversity
    ranked = _mmr_order(idx_to_vec, idx_to_score, lambda_param=0.25)

    # 5️⃣ Collect primary + neighbours
    primary = ranked[:top_n]
    neighbour_idxs = {
        j
        for i in primary
        for j in range(i - neighbor_window, i + neighbor_window + 1)
        if 0 <= j < len(idx_to_text)
    }
    final_order = [i for i in ranked if i in neighbour_idxs]

    # 6️⃣ Deduplicate & add metadata prefix
    seen: set[str] = set()
    docs: List[str] = []
    for i in final_order:
        raw = idx_to_text[i].strip()
        key = raw[:120]
        if key in seen:
            continue
        seen.add(key)
        meta = hits[i].payload or {}
        title = meta.get("title") or meta.get("source")
        prefix = f"### {title.split('/')[-1]}\n" if title else ""
        docs.append(prefix + raw)

    logger.debug(
        "📚 Returning %d unique docs. Combined chars=%d",
        len(docs),
        sum(len(d) for d in docs),
    )
    return docs


# ------------------------------------------------------------------
# Prompt construction (unchanged)
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# History-trimming helper (unchanged)
# ------------------------------------------------------------------


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
