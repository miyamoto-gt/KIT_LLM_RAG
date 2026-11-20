"""④ ベクトル索引の作成と、⑥ 検索（ベクトル＋任意でBM25のハイブリッド）。"""
import os
import re

import numpy as np

from config import CFG
from schema import Chunk, load_chunks, save_chunks

MAX_PER_SOURCE = getattr(CFG, "max_per_source", 1)     # 同じページから取るチャンク数の上限
EXPAND_NEIGHBORS = getattr(CFG, "expand_neighbors", 1)  # 前後に何チャンクつなげるか（0で無効）
PRIORITY_TYPES = ("manual", "map")                      # 手作りの正確なデータ
PRIORITY_TOP = getattr(CFG, "priority_top", 2)          # 手作りデータを最大何件まで優先するか
PRIORITY_MARGIN = getattr(CFG, "priority_margin", 0.05)  # 一番近いWebチャンクとの類似度の差がこれ以内なら優先して入れる
CURRENT_YEAR = getattr(CFG, "current_year", 2026)       # 新しさの基準年
RECENCY_DECAY = getattr(CFG, "recency_decay", 0.9)      # 1年古くなるごとに点数を何倍にするか（1.0で無効）
MAX_AGE = 5                                             # これ以上古くても下げ幅は同じ

_URL_YEAR = re.compile(r"(?<!\d)(20[0-3]\d)(?=\d{4}(?!\d)|\D|$)")   # /2024/ や 20231104_ など
_URL_REIWA = re.compile(r"(?<![a-z])r(\d{1,2})_")         # r8_kougaku など（令和）
_TEXT_YEAR = re.compile(r"(20[0-3]\d)\s*年")
_TEXT_REIWA = re.compile(r"令和\s*(\d{1,2}|元)\s*年")


def chunk_year(c: Chunk):
    """URLと本文の先頭から、そのチャンクの年を推定する（一番新しい年を採用）。見つからなければ None。"""
    years = [int(y) for y in _URL_YEAR.findall(c.source)]
    years += [2018 + int(n) for n in _URL_REIWA.findall(c.source)]
    head = c.text[:300]
    years += [int(y) for y in _TEXT_YEAR.findall(head)]
    years += [2018 + (1 if n == "元" else int(n)) for n in _TEXT_REIWA.findall(head)]
    years = [y for y in years if 1990 <= y <= CURRENT_YEAR + 2]
    return max(years) if years else None


def recency_weight(c: Chunk) -> float:
    """古いチャンクほど小さい倍率。手作りデータと年の分からないチャンクは 1.0。"""
    if c.type in PRIORITY_TYPES:
        return 1.0
    y = chunk_year(c)
    if y is None:
        return 1.0
    age = min(max(CURRENT_YEAR - y, 0), MAX_AGE)
    return RECENCY_DECAY ** age


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
        self.recency = np.array([recency_weight(c) for c in self.chunks], dtype="float32")
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
        if EXPAND_NEIGHBORS <= 0 or self.chunks[i].type in PRIORITY_TYPES:   # 手作りデータは1件で完結
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
            base = dict(dense)
        else:   # Reciprocal Rank Fusion でベクトル順位とキーワード順位を統合
            scores = self.bm25.get_scores(char_bigrams(query))
            sparse_rank = [int(i) for i in np.argsort(-scores)[:pool]]
            base = {}
            for rank_list in (dense_rank, sparse_rank):
                for r, i in enumerate(rank_list):
                    base[i] = base.get(i, 0.0) + 1.0 / (60 + r)
        # 古い情報ほど点数を下げて並べ直す（消すわけではない）
        order = sorted(base, key=lambda i: base[i] * self.recency[i], reverse=True)

        # 手作りデータで十分近いものがあれば、先頭に入れて必ず資料に含める
        if self.prio_vecs is not None and dense:
            s = self.prio_vecs @ q[0]
            line = max(dense.values()) - PRIORITY_MARGIN
            best = [int(j) for j in np.argsort(-s)[:PRIORITY_TOP] if s[j] >= line]
            picked = [self.prio_ids[j] for j in best]
            for j in best:
                dense[self.prio_ids[j]] = float(s[j])
            order = picked + [i for i in order if i not in picked]

        out, per_source = [], {}
        for i in order:
            c = self.chunks[i]
            src = c.id if c.type in PRIORITY_TYPES else c.source   # 手作りデータは1件ずつ別扱い
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