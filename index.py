"""④ ベクトル索引の作成と、⑥ 検索（ベクトル＋任意でBM25のハイブリッド）。"""
import os

import numpy as np

from config import CFG
from schema import Chunk, load_chunks, save_chunks

MAX_PER_SOURCE = getattr(CFG, "max_per_source", 1)     # 同じページから取るチャンク数の上限
EXPAND_NEIGHBORS = getattr(CFG, "expand_neighbors", 1)  # 前後に何チャンクつなげるか（0で無効）
PRIORITY_TYPES = ("manual", "map")                      # 手作りの正確なデータ
PRIORITY_MARGIN = getattr(CFG, "priority_margin", 0.05)  # 一番近いWebチャンクとの類似度の差がこれ以内なら優先して入れる


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


def _body(text: str) -> str:
    """先頭の【タイトル】行を除いた本文。"""
    return text.split("\n", 1)[-1] if text.startswith("【") else text


class Retriever:
    """search(query) -> [(Chunk, ベクトル類似度)]。add() で回答時にチャンクを追加できる。"""

    def __init__(self, embedder: Embedder = None):
        import faiss
        self.embedder = embedder or Embedder()
        self.index = faiss.read_index(os.path.join(CFG.index_dir, "faiss.index"))
        self.chunks = load_chunks(os.path.join(CFG.index_dir, "chunks.jsonl"))
        self._build_bm25()

    def _build_bm25(self):
        # 手作りデータのベクトルを別に持っておき、毎回まとめて類似度を測る（件数が少ないので速い）
        self.prio_ids = [i for i, c in enumerate(self.chunks) if c.type in PRIORITY_TYPES]
        self.prio_vecs = (np.stack([self.index.reconstruct(i) for i in self.prio_ids])
                          if self.prio_ids else None)
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

    def _expand(self, i: int) -> Chunk:
        """同じページの前後のチャンクをつなげて、途中で切れた一覧などを補う。"""
        if EXPAND_NEIGHBORS <= 0:
            return self.chunks[i]
        src = self.chunks[i].source
        lo = i
        while lo > 0 and i - lo < EXPAND_NEIGHBORS and self.chunks[lo - 1].source == src:
            lo -= 1
        hi = i
        while hi + 1 < len(self.chunks) and hi - i < EXPAND_NEIGHBORS and self.chunks[hi + 1].source == src:
            hi += 1
        if lo == hi:
            return self.chunks[i]
        base = self.chunks[i]
        header = base.text.split("\n", 1)[0] if base.text.startswith("【") else ""
        body = "\n".join(_body(self.chunks[j].text) for j in range(lo, hi + 1))
        return Chunk(id=base.id, text=f"{header}\n{body}" if header else body, source=src,
                     title=base.title, breadcrumb=base.breadcrumb, type=base.type)

    def search(self, query: str, k: int = None) -> list:
        k = k or CFG.top_k
        pool = k * 10
        q = self.embedder.query(query)
        sims, ids = self.index.search(q, pool)
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

        # 手作りデータで十分近いものがあれば、先頭に入れて必ず資料に含める
        if self.prio_vecs is not None and dense:
            s = self.prio_vecs @ q[0]
            j = int(np.argmax(s))
            if s[j] >= max(dense.values()) - PRIORITY_MARGIN:
                best = self.prio_ids[j]
                dense[best] = float(s[j])
                order = [best] + [i for i in order if i != best]

        out, per_source = [], {}
        for i in order:
            src = self.chunks[i].source
            if per_source.get(src, 0) >= MAX_PER_SOURCE:   # 同じページばかりで枠を埋めない
                continue
            per_source[src] = per_source.get(src, 0) + 1
            if i not in dense:     # BM25だけで拾ったものは類似度を個別計算
                dense[i] = float(self.index.reconstruct(i) @ q[0])
            out.append((self._expand(i), dense[i]))
            if len(out) >= k:
                break
        return out


if __name__ == "__main__":
    build_index()