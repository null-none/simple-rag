from sentence_transformers import SentenceTransformer
from typing import List


class Embedding:
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(text).tolist() for text in texts]

    def embed_query(self, query: str) -> List[float]:
        encoded_query = self.model.encode(query)
        return encoded_query.tolist()
