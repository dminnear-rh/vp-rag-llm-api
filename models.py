from typing import List, Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    """
    Represents a RAG (Retrieval-Augmented Generation) request to the API.

    Attributes:
        question (str): The current user question to be answered by the model.
        history (List[str]): A list of alternating user/assistant messages, in chronological order.
            Must alternate: user → assistant → user → assistant → ...
        model (Optional[str]): Optional override to specify a non-default model by name.
            If omitted, the system's default model will be used.
    """

    question: str
    history: List[str] = []
    model: Optional[str] = None  # Optional model override
