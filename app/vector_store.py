import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension=None):

        if dimension is not None:
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = None

    # =========================================================
    # ADD EMBEDDINGS
    # =========================================================

    def add(self, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if self.index is None:

            dimension = embeddings.shape[1]

            self.index = faiss.IndexFlatIP(
                dimension
            )

        # Normalize document embeddings.
        # Inner product of normalized vectors = cosine similarity.
        faiss.normalize_L2(
            embeddings
        )

        self.index.add(
            embeddings
        )

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        if self.index is None:

            print(
                "VectorStore: FAISS index is not initialized."
            )

            return []

        if self.index.ntotal == 0:

            print(
                "VectorStore: FAISS index is empty."
            )

            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        if query_embedding.ndim == 1:

            query_embedding = (
                query_embedding.reshape(1, -1)
            )

        if query_embedding.shape[1] != self.index.d:

            raise ValueError(
                f"Query embedding dimension "
                f"{query_embedding.shape[1]} does not match "
                f"FAISS index dimension {self.index.d}."
            )

        # Normalize query embedding.
        faiss.normalize_L2(
            query_embedding
        )

        actual_k = min(
            int(top_k),
            int(self.index.ntotal)
        )

        scores, indices = self.index.search(
            query_embedding,
            actual_k
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            results.append(
                {
                    "index": int(index),
                    "score": float(score)
                }
            )

        print(
            f"VectorStore: searched {self.index.ntotal} "
            f"vectors, returned {len(results)} results."
        )

        if results:

            print(
                "VectorStore: Top FAISS scores:"
            )

            for result in results[:10]:

                print(
                    f"  index={result['index']} "
                    f"score={result['score']:.4f}"
                )

        return results

    # =========================================================
    # SAVE
    # =========================================================

    def save(self, path):

        if self.index is None:

            raise ValueError(
                "Cannot save an uninitialized FAISS index."
            )

        faiss.write_index(
            self.index,
            path
        )

    # =========================================================
    # LOAD
    # =========================================================

    @classmethod
    def load(cls, path):

        store = cls()

        store.index = faiss.read_index(
            path
        )

        print(
            f"VectorStore: Loaded FAISS index "
            f"with {store.index.ntotal} vectors."
        )

        print(
            f"VectorStore: Dimension = "
            f"{store.index.d}"
        )

        return store

    # =========================================================
    # INFO
    # =========================================================

    def size(self):

        if self.index is None:
            return 0

        return int(
            self.index.ntotal
        )

    def dimension(self):

        if self.index is None:
            return 0

        return int(
            self.index.d
        )