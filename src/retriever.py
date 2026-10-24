from src.chunker import chunk_pages, Chunk
from src.embedder import embed_documents_cached, embed_query
from src.vector_store import VectorStore
from src.config import TOP_K, MIN_SCORE


class Retriever:
    def __init__(self, pages: list[dict]):
        self.chunks = chunk_pages(pages)
        vectors = embed_documents_cached([c.text for c in self.chunks])
        self.store = VectorStore(dim=vectors.shape[1])
        self.store.add(self.chunks, vectors)

    def retrieve(
        self, query: str, k: int = TOP_K, min_score: float = MIN_SCORE
    ) -> list[tuple[Chunk, float]]:
        results = self.store.search(embed_query(query), k=k)
        return [(c, s) for c, s in results if s >= min_score]
