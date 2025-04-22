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
    top_n: int = 5,
    neighbor_window: int = 2,
) -> List[str]:
    logger.debug(
        "🔍 get_context(coll=%r, limit=%d, doc_k=%d, win=%d) - %r",
        config.qdrant_collection,
        limit,
        top_n,
        neighbor_window,
        query,
    )

    qclient = QdrantClient(url=config.qdrant_url)
    q_vec = config.embedder.encode(f"query: {query}", normalize_embeddings=True)
    hits = qclient.query_points(
        collection_name=config.qdrant_collection,
        query=q_vec.tolist(),
        limit=limit,
        with_payload=True,
    ).points
    if not hits:
        return []

    # 1️⃣ Cross‑encoder re‑rank (optional)
    scores = [h.score for h in hits]
    if getattr(config, "cross_encoder", None):
        scores = config.cross_encoder.predict(
            [(query, h.payload["page_content"]) for h in hits]
        )
    ranked = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)

    # 2️⃣ Pick top‑N primary chunks
    primaries = [hits[i] for i in ranked[:top_n]]

    # 3️⃣ Expand ±window neighbours by index, then merge overlaps
    neighbour_pts: list[ScoredPoint] = []
    for p in primaries:
        src, cid = p.payload["source"], p.payload["chunk_id"]
        lo, hi = cid - neighbor_window, cid + neighbor_window
        pts, _ = qclient.scroll(
            collection_name=config.qdrant_collection,
            scroll_filter={
                "must": [
                    {"key": "source", "match": {"value": src}},
                    {"key": "chunk_id", "range": {"gte": lo, "lte": hi}},
                ]
            },
            with_payload=True,
        )
        neighbour_pts.extend(pts)

    # 4️⃣ Sort ⇒ iterate, merging consecutive chunks that overlap
    neighbour_pts.sort(key=lambda p: (p.payload["source"], p.payload["chunk_id"]))

    merged_docs: list[str] = []
    buf, last_src, last_id = [], None, None
    for p in neighbour_pts:
        src, cid, txt = (p.payload[k] for k in ("source", "chunk_id", "page_content"))
        if src == last_src and cid == last_id + 1:  # ‑‑ contiguous
            #  tiny heuristic: drop prefix already present (overlap)
            overlap = min(len(txt), config.chunk_overlap)
            buf.append(txt[overlap:].lstrip())
        else:
            if buf:
                merged_docs.append("\n".join(buf).strip())
            buf = [txt]
        last_src, last_id = src, cid
    if buf:
        merged_docs.append("\n".join(buf).strip())

    token_total = count_tokens("\n".join(merged_docs))
    logger.debug(
        "📚 %d merged docs | tokens=%d | chars=%d",
        len(merged_docs),
        token_total,
        sum(len(d) for d in merged_docs),
    )
    return merged_docs


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
