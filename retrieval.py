from typing import List

from qdrant_client import QdrantClient

from config import AppConfig


def get_context(
    config: AppConfig, query: str, limit: int = 20, top_n: int = 5
) -> List[str]:
    qdrant = QdrantClient(url=config.qdrant_url)
    embedded_query = config.embedder.encode(
        f"query: {query}", normalize_embeddings=True
    ).tolist()

    search_result = qdrant.search(
        collection_name=config.qdrant_collection,
        query_vector=embedded_query,
        limit=limit,
    )
    docs = [hit.payload.get("text", "") for hit in search_result]
    pairs = [(query, doc) for doc in docs]
    scores = config.cross_encoder.predict(pairs)
    reranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in reranked[:top_n]]


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

    return trim_history_for_tokens(base, history, question, max_tokens=25000)


def trim_history_for_tokens(
    messages: list[dict], history: list[str], question: str, max_tokens: int
) -> list[dict]:
    from utils import count_tokens

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
