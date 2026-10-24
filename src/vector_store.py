import faiss
import numpy as np
from src.chunker import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)   # inner product
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        k = min(k, len(self.chunks))
        scores, ids = self.index.search(query_vec, k)
        return [
            (self.chunks[i], float(s))
            for i, s in zip(ids[0], scores[0])
            if i != -1
        ]
