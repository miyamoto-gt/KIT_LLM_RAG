"""検索：埋め込み（API）＋ 文字bigram の BM25 を順位融合（RRF）し、これまでの工夫を足す。

  ・古い年の資料ほど点数を下げる（URL と本文の先頭から年を推定。手作りデータは対象外）
  ・同じページからは MAX_PER_SOURCE 件まで
  ・手作りデータで十分近いものは先頭に入れる
  ・（KEEP_TOP1）ベクトル・BM25 それぞれの1位を必ず残す
  ・前後のチャンクをつないで、途中で切れた一覧などを補う

資料の埋め込みは起動時に API で計算する（並列・時間制限つき）。時間内に終わらなければ BM25 だけで検索する。
手元で何度も動かすときは ~/.cache/kitrag/ に保存して使い回す（提出の zip には入らない）。
"""
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import llm
from . import settings as S

_URL_YEAR = re.compile(r"(?<!\d)(20[0-3]\d)(?=\d{4}(?!\d)|\D|$)")
_URL_REIWA = re.compile(r"(?<![a-z])r(\d{1,2})_")
_TEXT_YEAR = re.compile(r"(20[0-3]\d)\s*年")
_TEXT_REIWA = re.compile(r"令和\s*(\d{1,2}|元)\s*年")


def chunk_year(c):
    years = [int(y) for y in _URL_YEAR.findall(c.source)]
    years += [2018 + int(n) for n in _URL_REIWA.findall(c.source)]
    head = c.text[:300]
    years += [int(y) for y in _TEXT_YEAR.findall(head)]
    years += [2018 + (1 if n == "元" else int(n)) for n in _TEXT_REIWA.findall(head)]
    years = [y for y in years if 1990 <= y <= S.CURRENT_YEAR + 2]
    return max(years) if years else None


def recency_weight(c):
    if c.type == "manual":
        return 1.0
    y = chunk_year(c)
    if y is None:
        return 1.0
    return S.RECENCY_DECAY ** min(max(S.CURRENT_YEAR - y, 0), S.MAX_AGE)


def char_bigrams(text):
    t = "".join(text.split())
    return [t[i:i + 2] for i in range(len(t) - 1)] or [t]


class _SimpleBM25:
    """rank_bm25 が無いとき用の同じ式の実装。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        from collections import Counter
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter(g for d in self.tf for g in d)
        n = len(docs)
        self.idf = {g: np.log(1 + (n - c + 0.5) / (c + 0.5)) for g, c in df.items()}
        self.k1, self.b = k1, b

    def get_scores(self, q):
        s = np.zeros(len(self.tf), dtype="float32")
        for g in set(q):
            if g not in self.idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(g)
                if f:
                    s[i] += self.idf[g] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


class Retriever:
    def __init__(self, chunks, log=print):
        self.chunks = chunks
        self.log = log
        self.recency = np.array([recency_weight(c) for c in chunks], dtype="float32")
        self.prio_ids = [c.id for c in chunks if c.type == "manual"]
        docs = [char_bigrams(c.text) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(docs)
        except Exception:
            self.bm25 = _SimpleBM25(docs)
        self.vectors = self._embed_all()                    # None なら BM25 だけ
        # 同じページの前後のチャンク（source と seq で引く）
        self.by_source = {}
        for c in chunks:
            self.by_source.setdefault(c.source, {})[c.seq] = c

    # ---- 資料の埋め込み ----
    def _cache_path(self):
        h = hashlib.sha256((S.EMBEDDING_MODEL + "\n".join(c.text for c in self.chunks)).encode()).hexdigest()[:16]
        return os.path.join(os.path.expanduser("~"), ".cache", "kitrag", f"emb_{h}.npy")

    def _embed_all(self):
        if not self.chunks or not S.LLM_BASE_URL and not os.environ.get("OPENAI_BASE_URL"):
            self.log("埋め込み API が無いので BM25 だけで検索します")
            return None
        self.log(f"埋め込みモデル: {S.EMBEDDING_MODEL}（質問「{S.QUERY_PREFIX}」・資料「{S.DOC_PREFIX}」）")
        cache = self._cache_path()
        try:
            if os.path.exists(cache):
                v = np.load(cache)
                if v.shape[0] == len(self.chunks):
                    self.log(f"埋め込みをキャッシュから読みました（{cache}）")
                    return v
        except Exception:
            pass
        t0 = time.time()
        texts = [S.DOC_PREFIX + c.text for c in self.chunks]
        starts = list(range(0, len(texts), S.EMBED_BATCH))
        out = [None] * len(texts)
        fails = [0]

        def run(lo, hi):
            """lo〜hi を埋め込む。大きいまとまりで失敗したら、すぐ半分に分けてやり直す。
            4件以下でも失敗するときだけ、待ってやり直す（サーバーが一時的に落ちている場合）。"""
            if time.time() - t0 > S.EMBED_TIME_LIMIT or fails[0] > 20:   # 時間切れ・サーバーが落ちている
                return
            small = hi - lo <= 4
            try:
                out[lo:hi] = llm.embed(texts[lo:hi], tries=3 if small else 1)
            except Exception as e:
                if small:
                    fails[0] += 1
                    if fails[0] <= 3:
                        self.log(f"埋め込みに失敗（{hi - lo}件）: {type(e).__name__}: {str(e)[:80]}")
                    return
                mid = (lo + hi) // 2
                run(lo, mid)
                run(mid, hi)

        with ThreadPoolExecutor(max_workers=S.EMBED_WORKERS) as pool:
            list(pool.map(lambda st: run(st, min(st + S.EMBED_BATCH, len(texts))), starts))
        missing = sum(o is None for o in out)
        if missing > 0.3 * len(texts):
            self.log(f"埋め込みが {missing}/{len(texts)} 件できなかったので BM25 だけで検索します（{time.time() - t0:.0f}秒）")
            return None
        dim = len(next(o for o in out if o is not None))
        out = [o if o is not None else [0.0] * dim for o in out]   # できなかったチャンクは BM25 だけで拾う
        if missing:
            self.log(f"埋め込みが {missing} 件できず、その分は BM25 だけで検索します")
        v = np.asarray(out, dtype="float32")
        self.log(f"資料 {len(texts)} チャンクの埋め込み：{time.time() - t0:.0f}秒")
        if missing:
            return v                                   # 欠けがあるものはキャッシュしない
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            np.save(cache, v)
        except Exception:
            pass
        return v

    # ---- 検索 ----
    def _expand(self, c):
        if S.EXPAND_NEIGHBORS <= 0 or c.type == "manual":
            return c
        page = self.by_source.get(c.source, {})
        seqs = [s for s in range(c.seq - S.EXPAND_NEIGHBORS, c.seq + S.EXPAND_NEIGHBORS + 1) if s in page]
        if len(seqs) <= 1:
            return c
        body = ""
        for s in seqs:
            t = page[s].text
            t = t.split("\n", 1)[1] if c.title and t.startswith(c.title) and "\n" in t else t
            # 分割の重なり（overlap）を取り除いてつなぐ
            k = next((k for k in range(min(S.CHUNK_OVERLAP, len(body), len(t)), 10, -1) if body.endswith(t[:k])), 0)
            body = body + t[k:] if body else t
        from dataclasses import replace
        return replace(c, text=f"{c.title}\n{body}" if c.title else body)

    def search(self, query, k=S.TOP_K, use_dense=True, debug=False):
        """debug=True なら (結果, {"dense": 候補の番号, "bm25": 候補の番号}) を返す（評価用）。"""
        n = len(self.chunks)
        if n == 0:
            return []
        dense, dense_rank = {}, []
        q = None
        if self.vectors is not None and use_dense:
            try:
                q = np.asarray(llm.embed([S.QUERY_PREFIX + query], tries=2)[0], dtype="float32")
                sims = self.vectors @ q
                dense_rank = [int(i) for i in np.argsort(-sims)[:S.POOL]]
                dense = {i: float(sims[i]) for i in dense_rank}
            except Exception as e:
                self.log(f"質問の埋め込みに失敗（BM25 だけで検索）: {type(e).__name__}")
                q = None
        scores = np.asarray(self.bm25.get_scores(char_bigrams(query)), dtype="float32")
        sparse_rank = [int(i) for i in np.argsort(-scores)[:S.POOL]]

        base = {}
        for rl in (dense_rank, sparse_rank):
            for r, i in enumerate(rl):
                base[i] = base.get(i, 0.0) + 1.0 / (60 + r)
        order = sorted(base, key=lambda i: base[i] * self.recency[i], reverse=True)

        picked = []
        if q is not None and self.prio_ids and dense:
            pv = self.vectors[self.prio_ids] @ q
            line = max(dense.values()) - S.PRIORITY_MARGIN
            picked = [self.prio_ids[j] for j in np.argsort(-pv)[:S.PRIORITY_TOP] if pv[j] >= line]
            order = picked + [i for i in order if i not in picked]
        if S.KEEP_TOP1:
            must = []
            for rl in (dense_rank[:1], sparse_rank[:1]):
                must += [i for i in rl if i not in picked and i not in must]
            must = must[:max(0, k - len(picked) - 1)]
            order = picked + must + [i for i in order if i not in picked and i not in must]

        out, per_source = [], {}
        for i in order:
            c = self.chunks[i]
            key = c.id if c.type == "manual" else c.source
            if per_source.get(key, 0) >= S.MAX_PER_SOURCE:
                continue
            per_source[key] = per_source.get(key, 0) + 1
            out.append((self._expand(c), dense.get(i, 0.0)))
            if len(out) >= k:
                break
        if debug:
            return out, {"dense": dense_rank, "bm25": sparse_rank}
        return out
