import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from qdrant_client import QdrantClient

from config import AppConfig
from utils import count_tokens

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers: cosine + Max-Marginal Relevance
# ------------------------------------------------------------------
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _mmr(
    vecs: Dict[str, np.ndarray],
    scores: Dict[str, float],
    k: int,
    λ: float = 0.25,
) -> List[str]:
    """Return *document IDs* ordered by Max-Marginal Relevance."""
    picked: List[str] = []
    cand = set(vecs)

    while cand and len(picked) < k:
        if not picked:
            best = max(cand, key=lambda d: scores[d])
        else:
            best = max(
                cand,
                key=lambda d: λ * scores[d]
                - (1 - λ) * max(_cos(vecs[d], vecs[p]) for p in picked),
            )
        picked.append(best)
        cand.remove(best)
    return picked


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------
def get_context(
    config: AppConfig,
    query: str,
    limit: int = 100,  # raw vector hits
    doc_k: int = 4,  # how many *documents* to keep
    max_ctx_tokens: int = 10_000,
) -> List[str]:
    logger.debug(
        "🔍 get_context(collection=%r, limit=%d, doc_k=%d) - %r",
        config.qdrant_collection,
        limit,
        doc_k,
        query,
    )

    # 1️⃣ embed query
    q_embed = config.embedder.encode(f"query: {query}", normalize_embeddings=True)

    # 2️⃣ similarity search
    client = QdrantClient(url=config.qdrant_url)
    hits = client.search(
        collection_name=config.qdrant_collection,
        query_vector=q_embed.tolist(),
        limit=limit,
    )
    if not hits:
        logger.warning("⚠️  No hits from Qdrant")
        return []

    # 3️⃣ build structures grouped by *document* id
    doc_texts: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    doc_vecs: Dict[str, np.ndarray] = {}
    doc_scores: Dict[str, float] = {}

    for global_idx, hit in enumerate(hits):
        meta = hit.payload or {}
        doc_id = meta.get("source") or meta.get("title") or f"point-{hit.id}"
        text = (
            meta.get("text")
            or meta.get("content")
            or meta.get("chunk")
            or " ".join(v for v in meta.values() if isinstance(v, str))
        )

        doc_texts[doc_id].append((global_idx, text))
        # store the *first* vector / score we see for the doc
        if doc_id not in doc_vecs:
            doc_vecs[doc_id] = np.asarray(hit.vector, dtype=np.float32)
            doc_scores[doc_id] = hit.score

    # 4️⃣ optional cross-encoder rescoring
    if getattr(config, "cross_encoder", None):
        ce_pairs = [
            (query, " ".join(t for _, t in doc_texts[d][:3])) for d in doc_texts
        ]
        ce_scores = config.cross_encoder.predict(ce_pairs)
        for doc_id, s in zip(doc_texts, ce_scores):
            doc_scores[doc_id] = float(s)

    # 5️⃣ MMR to pick K diverse documents
    pick_ids = _mmr(doc_vecs, doc_scores, k=doc_k, λ=0.25)

    # 6️⃣ merge all chunks per doc (ordered) and respect token budget
    results: List[str] = []
    total_tokens = 0
    for d in pick_ids:
        # sort by original order
        body = "\n".join(t for _, t in sorted(doc_texts[d]))
        title = d.split("/")[-1]
        passage = f"### {title}\n{body}".strip()

        tokens_here = count_tokens([{"role": "user", "content": passage}])
        if total_tokens + tokens_here > max_ctx_tokens:
            break
        results.append(passage)
        total_tokens += tokens_here

    logger.debug(
        "📚 Selected %d docs | tokens=%d | chars=%d",
        len(results),
        total_tokens,
        sum(len(r) for r in results),
    )
    return results


# ------------------------------------------------------------------
# Prompt builder
# ------------------------------------------------------------------
def build_messages(
    question: str,
    retrieved_docs: List[str],
    history: List[str],
    *,
    max_prompt_tokens: int = 25_000,
) -> List[dict]:
    context = "\n---\n".join(retrieved_docs)

    system_msg = (
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
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"Context:\n{context}"},
    ]
    return _trim_for_tokens(base, history, question, max_prompt_tokens)


# ------------------------------------------------------------------
# History-trimmer
# ------------------------------------------------------------------
def _trim_for_tokens(
    messages: List[dict],
    history: List[str],
    question: str,
    max_tokens: int,
) -> List[dict]:
    messages = messages.copy()
    for turn in history:
        messages.append({"role": "user", "content": turn})
        if (
            count_tokens(
                messages + [{"role": "user", "content": f"Question: {question}"}]
            )
            > max_tokens
        ):
            messages.pop(2)  # drop oldest context block
            break
    messages.append({"role": "user", "content": f"Question: {question}"})
    return messages
