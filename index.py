"""④ ベクトル索引の作成と、⑥ 検索（ベクトル＋任意でBM25のハイブリッド）。"""
import os

import numpy as np

from KIT_LLM_RAG.config import CFG
from schema import load_chunks, save_chunks


def char_bigrams(text: str) -> list:
    """形態素解析なしで日本語のキーワード検索をするための文字bigram。"""
    t = "".join(text.split())
    return [t[i:i + 2] for i in range(len(t) - 1)] or [t]


class Embedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(CFG.embed_model)

    def docs(self, texts):
        return self.model.encode([CFG.doc_prefix + t for t in texts], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=True).astype("float32")

    def query(self, text):
        return self.model.encode([CFG.query_prefix + text],
                                 normalize_embeddings=True).astype("float32")


def build_index() -> None:
    import faiss
    chunks = load_chunks(CFG.chunks_path)
    vecs = Embedder().docs([c.text for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])     # 正規化済みなので内積＝コサイン類似度
    index.add(vecs)
    os.makedirs(CFG.index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(CFG.index_dir, "faiss.index"))
    save_chunks(chunks, os.path.join(CFG.index_dir, "chunks.jsonl"))  # 索引と同じ並び順で保存
    print(f"done: {len(chunks)} vectors, dim={vecs.shape[1]}")


class Retriever:
    """search(query) -> [(Chunk, ベクトル類似度)]。add() で回答時にチャンクを追加できる。"""

    def __init__(self, embedder: Embedder = None):
        import faiss
        self.embedder = embedder or Embedder()
        self.index = faiss.read_index(os.path.join(CFG.index_dir, "faiss.index"))
        self.chunks = load_chunks(os.path.join(CFG.index_dir, "chunks.jsonl"))
        self._build_bm25()

    def _build_bm25(self):
        self.bm25 = None
        if CFG.use_bm25:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([char_bigrams(c.text) for c in self.chunks])

    def add(self, new_chunks: list) -> None:
        if not new_chunks:
            return
        self.index.add(self.embedder.docs([c.text for c in new_chunks]))
        self.chunks.extend(new_chunks)
        self._build_bm25()

    def search(self, query: str, k: int = None) -> list:
        k = k or CFG.top_k
        pool = k * 5
        sims, ids = self.index.search(self.embedder.query(query), pool)
        dense = {int(i): float(s) for i, s in zip(ids[0], sims[0]) if i >= 0}
        dense_rank = list(dense)

        if self.bm25 is None:
            order = dense_rank
        else:   # Reciprocal Rank Fusion でベクトル順位とキーワード順位を統合
            scores = self.bm25.get_scores(char_bigrams(query))
            sparse_rank = [int(i) for i in np.argsort(-scores)[:pool]]
            fused = {}
            for rank_list in (dense_rank, sparse_rank):
                for r, i in enumerate(rank_list):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (60 + r)
            order = sorted(fused, key=fused.get, reverse=True)

        q = None
        out = []
        for i in order[:k]:
            if i not in dense:     # BM25だけで拾ったものは類似度を個別計算
                q = self.embedder.query(query) if q is None else q
                dense[i] = float(self.index.reconstruct(i) @ q[0])
            out.append((self.chunks[i], dense[i]))
        return out


if __name__ == "__main__":
    build_index()
