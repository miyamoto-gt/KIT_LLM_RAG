"""OpenAI 互換 API（サンドボックスの DGX 上のローカル LLM）を呼ぶ薄い部品。"""
import math
import time

from . import settings as S

_client = None
STATS = []                      # 生成1回にかかった秒数（混み具合の目安）


def client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(base_url=S.LLM_BASE_URL or None, api_key=S.API_KEY, timeout=S.REQUEST_TIMEOUT,
                         max_retries=0)       # やり直しは _retry だけで行う（ライブラリ側でも2回やり直すと待ち時間が3倍になる）
    return _client


def _timeout():
    """1回の呼び出しの待ち時間。締め切り（TIME_LIMIT）を越えて待たない。"""
    t = min(S.REQUEST_TIMEOUT, S.remaining() - 5)
    if t < 3:
        raise TimeoutError("締め切りが近いので API を呼びません")
    return t


def _retry(fn, tries=4):
    for k in range(tries):
        try:
            return fn(_timeout())
        except TimeoutError:
            raise
        except Exception as e:
            if "Timeout" in type(e).__name__:        # 混雑で待ちきれなかった：やり直すと列の最後に並び直すだけなので諦める
                raise
            if getattr(e, "status_code", None) == 429:   # 回数制限：やり直しても回数を使うだけ
                raise
            if k == tries - 1 or S.remaining() < 30:
                raise
            time.sleep(2.0 * (k + 1))


def chat(messages, max_tokens=128):
    t0 = time.time()
    r = _retry(lambda t: client().with_options(timeout=t).chat.completions.create(
        model=S.LLM_MODEL, messages=messages, temperature=0, max_tokens=max_tokens), tries=2)
    STATS.append(time.time() - t0)
    return (r.choices[0].message.content or "").strip()


def embed(texts, tries=4, model=None):
    """正規化したベクトル（list[list[float]]）を返す。"""
    r = _retry(lambda t: client().with_options(timeout=t).embeddings.create(model=model or S.EMBEDDING_MODEL, input=list(texts)),
               tries=tries)
    out = []
    for item in r.data:
        v = item.embedding
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out