import logging
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import ScoredPoint

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


def get_context(
    config: AppConfig,
    query: str,
    limit: int = 60,
    neighbor_window: int = 1,
) -> List[str]:
    logger.debug(
        "🔍 get_context(coll=%r, limit=%d, win=%d) - %r",
        config.qdrant_collection,
        limit,
        neighbor_window,
        query,
    )

    qclient = QdrantClient(url=config.qdrant_url)
    q_vec = config.embedder.encode(f"query: {query}", normalize_embeddings=True)
    hits = qclient.query_points(
        collection_name=config.qdrant_collection,
        query=q_vec.tolist(),
        limit=limit,
    ).points
    if not hits:
        return []

    # 1️⃣  Cross‑encoder re‑rank
    scores = [h.score for h in hits]
    if getattr(config, "cross_encoder", None):
        scores = config.cross_encoder.predict(
            [(query, h.payload["page_content"]) for h in hits]
        )
    ranked = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)

    # 2️⃣ Pick the best chunk from up‑to three different sources
    PRIMARY_PER_SOURCE = 3  # how many distinct sources we want
    primaries: list[ScoredPoint] = []
    used_sources: set[str] = set()

    for i in ranked:
        p = hits[i]
        src = p.payload["metadata"]["source"]
        if src not in used_sources:
            primaries.append(p)
            used_sources.add(src)
        if len(primaries) == PRIMARY_PER_SOURCE:
            break
    if not primaries:  # should never happen, but be safe
        return []

    # 3️⃣ For *each* primary, grab ±window neighbour chunks from the same source
    neighbour_pts: list[ScoredPoint] = []
    for p in primaries:
        meta = p.payload["metadata"]
        src, cid = meta["source"], meta["chunk_id"]
        lo, hi = cid - neighbor_window, cid + neighbor_window

        pts, _ = qclient.scroll(
            collection_name=config.qdrant_collection,
            scroll_filter={
                "must": [
                    {"key": "metadata.source", "match": {"value": src}},
                    {"key": "metadata.chunk_id", "range": {"gte": lo, "lte": hi}},
                ]
            },
            with_payload=True,
            with_vectors=False,
        )
        neighbour_pts.extend(pts)

    # 4️⃣ Order by (source, chunk_id) and dedupe by text
    neighbour_pts.sort(
        key=lambda p: (
            p.payload["metadata"]["source"],
            p.payload["metadata"]["chunk_id"],
        )
    )
    docs, seen = [], set()
    for p in neighbour_pts:
        txt = p.payload["page_content"].strip()
        if txt and txt not in seen:
            seen.add(txt)
            docs.append(txt)
            if len(docs) >= (PRIMARY_PER_SOURCE * (2 * neighbor_window + 1)):
                break  # don't exceed 9 if window=1

    return docs


def build_messages(
    question: str,
    retrieved_docs: List[str],
    history: List[str],
    *,
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
