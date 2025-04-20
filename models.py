from typing import List, Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    question: str
    history: List[str] = []
    model: Optional[str] = None  # Optional model override
