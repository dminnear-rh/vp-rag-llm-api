import logging
from collections import defaultdict
from typing import Dict, List, Set

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


def get_context(
    config: AppConfig,
    query: str,
    limit: int = 60,
    primary_k: int = 10,
    neighbor_window: int = 1,
    per_source_max: int = 3,
    min_rel_score: float = 0.55,
    min_abs_score: float = 0.25,
) -> List[str]:
    """
    Retrieve up to *primary_k* highly-relevant chunks **plus** their ± *neighbor_window*
    neighbours, with light source-level diversity.

    Steps
    -----
    1. Dense-vector search → optional cross-encoder re-rank
    2. Score/diversity filtering ⇒ *primaries*
    3. Build a set of wanted (source, chunk_id ± window) pairs
    4. One `scroll()` per source using a `match any=[…]` filter
    5. Order by *(source, chunk_id)* and return unique texts
    """
    logger.debug(
        "🔍 get_context(coll=%s, k=%d, win=%d, limit=%d) – %r",
        config.qdrant_collection,
        primary_k,
        neighbor_window,
        limit,
        query,
    )

    # ── 1. Vector search ────────────────────────────────────────────────────
    qclient = QdrantClient(url=config.qdrant_url)
    q_vec = config.embedder.encode(f"query: {query}", normalize_embeddings=True)

    search = qclient.query_points(
        collection_name=config.qdrant_collection,
        query=q_vec.tolist(),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    ).points

    if not search:
        logger.warning("⚠️  No vector hits returned from Qdrant")
        return []

    # ── 2. Optional cross-encoder re-rank ───────────────────────────────────
    scores = [p.score for p in search]
    if getattr(config, "cross_encoder", None):
        scores = config.cross_encoder.predict(
            [(query, p.payload["page_content"]) for p in search]
        )

    ranked_idx = sorted(range(len(search)), key=lambda i: scores[i], reverse=True)
    top_score = scores[ranked_idx[0]]

    # ── 3. Pick primary chunks (score-gated & per-source capped) ─────────────
    primaries: List[ScoredPoint] = []
    per_src_counter: Dict[str, int] = {}

    for i in ranked_idx:
        p = search[i]
        src = p.payload["metadata"]["source"]

        if scores[i] < max(min_abs_score, top_score * min_rel_score):
            continue
        if per_src_counter.get(src, 0) >= per_source_max:
            continue

        primaries.append(p)
        per_src_counter[src] = per_src_counter.get(src, 0) + 1
        if len(primaries) == primary_k:
            break

    if not primaries:
        primaries.append(search[ranked_idx[0]])

    # ── 4. Build neighbour id-sets & fetch in one scroll per source ─────────
    per_src_idset: Dict[str, Set[int]] = defaultdict(set)
    for p in primaries:
        meta = p.payload["metadata"]
        src, cid = meta["source"], meta["chunk_id"]
        for n in range(cid - neighbor_window, cid + neighbor_window + 1):
            per_src_idset[src].add(n)

    neighbour_pts: List[ScoredPoint] = []
    for src, idset in per_src_idset.items():
        pts, _ = qclient.scroll(
            collection_name=config.qdrant_collection,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": src}},
                    {"key": "metadata.chunk_id", "match": {"any": list(idset)}},
                ]
            },
            with_payload=True,
            with_vectors=False,
        )
        neighbour_pts.extend(pts)

    # ── 5. Order (source, chunk_id) & return  ────────────────────────────────
    neighbour_pts.sort(
        key=lambda p: (
            p.payload["metadata"]["source"],
            p.payload["metadata"]["chunk_id"],
        )
    )

    docs = [p.payload["page_content"].strip() for p in neighbour_pts]
    return docs


def build_messages(
    question: str,
    retrieved_docs: List[str],
    history: List[str],
    max_prompt_tokens: int = 25_000,
) -> List[dict]:
    context = "\n---\n".join(retrieved_docs)

    system_msg = (
        "You are an expert in Red Hat OpenShift and Kubernetes application "
        "architecture, specializing in the design, implementation, and "
        "validation of OpenShift Validated Patterns.\n\n"
        "The user is seeking to better understand Validated Patterns — "
        "including how they are built, tested, used across industries, or "
        "extended for new use cases. Use the following documentation and "
        "source content to generate a clear and practical answer.\n\n"
        "Your response should:\n"
        "- Reference relevant concepts or implementation details from the provided context\n"
        "- Be technically accurate and aimed at architects, engineers, or contributors\n"
        "- Include examples or explanations using OpenShift-native tools and practices "
        "(e.g., GitOps, Operators, Pipelines, Secrets management)"
    )

    base = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n{context}"},
    ]
    return _trim_history(base, history, question, max_prompt_tokens)


def _trim_history(
    messages: List[dict],
    history: List[str],
    question: str,
    max_tokens: int,
) -> List[dict]:
    msgs = messages.copy()
    for turn in history:
        msgs.append({"role": "user", "content": turn})
        if (
            count_tokens(msgs + [{"role": "user", "content": f"Question: {question}"}])
            > max_tokens
        ):
            msgs.pop(2)
            break
    msgs.append({"role": "user", "content": f"Question: {question}"})
    return msgs
