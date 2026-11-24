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
        self._t_embed = time.time()
        self.vecs = {}                                      # モデル名 → ベクトル（できなかったモデルは入らない）
        for m in S.EMBED_MODELS:
            v = self._embed_all(m)
            if v is not None:
                self.vecs[m] = v
        self.vectors = self.vecs.get(S.EMBEDDING_MODEL)     # 手作りデータの優先に使う（無ければ優先しない）
        # 同じページの前後のチャンク（source と seq で引く）
        self.by_source = {}
        for c in chunks:
            self.by_source.setdefault(c.source, {})[c.seq] = c

    # ---- 資料の埋め込み ----
    def _cache_path(self, model):
        h = hashlib.sha256((model + "\n".join(c.text for c in self.chunks)).encode()).hexdigest()[:16]
        return os.path.join(os.path.expanduser("~"), ".cache", "kitrag", f"emb_{h}.npy")

    def _embed_all(self, model):
        if not self.chunks or not S.LLM_BASE_URL and not os.environ.get("OPENAI_BASE_URL"):
            self.log("埋め込み API が無いので BM25 だけで検索します")
            return None
        qp, dp = S.prefixes(model)
        self.log(f"埋め込みモデル: {model}（質問「{qp}」・資料「{dp}」）")
        cache = self._cache_path(model)
        try:
            if os.path.exists(cache):
                v = np.load(cache)
                if v.shape[0] == len(self.chunks):
                    self.log(f"埋め込みをキャッシュから読みました（{cache}）")
                    return v
        except Exception:
            pass
        t0 = time.time()
        texts = [dp + c.text for c in self.chunks]
        starts = list(range(0, len(texts), S.EMBED_BATCH))
        out = [None] * len(texts)
        fails = [0]

        stop = [None]                                  # サーバー側の理由で打ち切ったときの理由

        def run(lo, hi):
            """lo〜hi を埋め込む。
            ・送る量が原因らしい失敗（400 / 413 など）だけ、半分に分けて送り直す
            ・サーバー側の失敗（429 回数制限・5xx 停止・タイムアウト）は分けても直らず回数を無駄にするので、
              3回続いたら全体を打ち切って BM25 に任せる（429 はすぐ打ち切る）"""
            if stop[0] or time.time() - self._t_embed > S.EMBED_TIME_LIMIT:
                return
            try:
                out[lo:hi] = llm.embed(texts[lo:hi], tries=2, model=model)
                fails[0] = 0
            except Exception as e:
                code = getattr(e, "status_code", None)
                server_side = code == 429 or (code or 0) >= 500 or "Timeout" in type(e).__name__ \
                    or "Connection" in type(e).__name__
                if server_side and code != 429 and hi - lo > 8:   # 大きいまとまりの 5xx は量のせいかもしれない：8件まで分けて試す
                    mid = (lo + hi) // 2
                    run(lo, mid)
                    run(mid, hi)
                    return
                if server_side:
                    fails[0] += 1
                    self.log(f"埋め込みに失敗（{hi - lo}件）: {type(e).__name__} {code or ''}")
                    if code == 429 or fails[0] >= 3:
                        stop[0] = f"{type(e).__name__} {code or ''}"
                    return
                if hi - lo > 4:                        # 送る量の問題 → 半分にして送り直す
                    mid = (lo + hi) // 2
                    run(lo, mid)
                    run(mid, hi)
                else:
                    self.log(f"埋め込みに失敗（{hi - lo}件）: {type(e).__name__}: {str(e)[:80]}")

        with ThreadPoolExecutor(max_workers=S.EMBED_WORKERS) as pool:
            list(pool.map(lambda st: run(st, min(st + S.EMBED_BATCH, len(texts))), starts))
        missing = sum(o is None for o in out)
        if stop[0]:
            self.log(f"サーバー側のエラー（{stop[0]}）が続いたので埋め込みを打ち切りました")
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

    def search_manual(self, query, k=2):
        """手作りデータだけから、質問に近いものを k 件（答えを拒否したときの探し直し用）。"""
        if not self.prio_ids:
            return []
        scores = np.asarray(self.bm25.get_scores(char_bigrams(query)), dtype="float32")[self.prio_ids]
        rank = {}
        for r, j in enumerate(np.argsort(-scores)[:10]):
            rank[self.prio_ids[j]] = rank.get(self.prio_ids[j], 0.0) + 1.0 / (60 + r)
        if self.vectors is not None:
            try:
                qv = np.asarray(llm.embed([S.QUERY_PREFIX + query], tries=1)[0], dtype="float32")
                sims = self.vectors[self.prio_ids] @ qv
                for r, j in enumerate(np.argsort(-sims)[:10]):
                    rank[self.prio_ids[j]] = rank.get(self.prio_ids[j], 0.0) + 1.0 / (60 + r)
            except Exception:
                pass
        best = sorted(rank, key=lambda i: -rank[i])[:k]
        return [(self.chunks[i], 1.0) for i in best]

    _PHRASE = re.compile(r"[（(「『]([^）)」』]{2,15})[）)」』]")

    def _phrase_boost(self, query, order):
        """質問に（専願型）「…」のような、種類を指定する語があれば、それを含む候補を前に出す。

        「総合型選抜（専願型）」のように、似た資料（Uターン型・女子奨学生…）が多いときに、
        指定された種類のページを選ぶため。候補（上位 S.POOL）の中だけで並べ替える。
        """
        if not S.PHRASE_BOOST:
            return order
        phrases = [p for p in self._PHRASE.findall(query) if not re.fullmatch(r"[0-9０-９年度月日\s]+", p)]
        if not phrases:
            return order
        top = order[:S.POOL]
        norm = lambda t: re.sub(r"\s+", "", t)
        has = [i for i in top if all(norm(p) in norm(self.chunks[i].text) for p in phrases)]
        if not has or len(has) == len(top):
            return order
        return has + [i for i in order if i not in set(has)]

    def search(self, query, k=S.TOP_K, use_dense=True, debug=False):
        """debug=True なら (結果, {"dense": 候補の番号, "bm25": 候補の番号}) を返す（評価用）。"""
        n = len(self.chunks)
        if n == 0:
            return []
        dense, dense_rank, q = {}, [], None              # dense / q は優先判定用のモデル（EMBED_MODELS の先頭）
        ranks = []                                         # モデルごとの候補の順位
        for m, vec in (self.vecs.items() if use_dense else []):
            try:
                qv = np.asarray(llm.embed([S.prefixes(m)[0] + query], tries=2, model=m)[0], dtype="float32")
                sims = vec @ qv
                rl = [int(i) for i in np.argsort(-sims)[:S.POOL]]
                ranks.append(rl)
                if m == S.EMBEDDING_MODEL:
                    q, dense_rank, dense = qv, rl, {i: float(sims[i]) for i in rl}
            except Exception as e:
                self.log(f"質問の埋め込みに失敗（{m}）: {type(e).__name__}")
        scores = np.asarray(self.bm25.get_scores(char_bigrams(query)), dtype="float32")
        sparse_rank = [int(i) for i in np.argsort(-scores)[:S.POOL]]

        base = {}
        for rl in ranks + [sparse_rank]:
            for r, i in enumerate(rl):
                base[i] = base.get(i, 0.0) + 1.0 / (60 + r)
        past = bool(S.PAST_Q.search(query))                 # 過去を聞く質問は新しさの補正をしない
        order = sorted(base, key=lambda i: base[i] * (1.0 if past else self.recency[i]), reverse=True)
        order = self._phrase_boost(query, order)

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
            return out, {"dense": sorted({i for rl in ranks for i in rl}), "dense_each": ranks, "bm25": sparse_rank}
        return out