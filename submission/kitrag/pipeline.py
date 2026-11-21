"""質問1つに答える中心部品。RAG(chunks).answer(question) -> dict。

  振り分け（router.plan）→ 検索（Retriever）→ 資料を絞る（prompts.build_context）
  → 生成（API）→ 計算問題なら式を計算して答え直す → 後処理
LLM が失敗したら、絞った資料の一番関係する文を答えにする（「不明」は0点なので空にしない）。
"""
import re

from . import llm
from . import settings as S
from .postprocess import ensure_calc_values, postprocess
from .prompts import build_calc_answer_messages, build_context, build_messages
from .retriever import Retriever
from .router import plan
from .tools import run_calc_all
from .repair import decompose, repair_message, unit_missing


def extractive(ctx: str) -> str:
    """LLM が使えないときの答え：資料の最初の文（タイトルを除く）。"""
    for line in ctx.splitlines():
        body = re.sub(r"^\[\d+\]\s*(【[^】]*】[^ ]*\s*)?", "", line).strip()
        if body:
            parts = re.split(r"(?<=。)|／", body)
            return (parts[0] if parts else body).strip()[:200]
    return S.FALLBACK


class RAG:
    def __init__(self, chunks, log=print):
        self.retriever = Retriever(chunks, log=log)
        self.log = log

    def answer(self, question: str, hits=None, split=True) -> dict:
        """hits を渡すと検索をとばしてその資料で答える（評価で「正解の資料だけ」を渡すとき用）。"""
        subs = decompose(question) if (split and hits is None and S.DECOMPOSE) else []
        if subs:                                        # 「AとBは、それぞれ〜」→ 1つずつ検索・生成してつなぐ
            outs = [self.answer(sq, split=False) for sq in subs]
            ans = "".join(o["answer"] if o["answer"].endswith("。") else o["answer"] + "。" for o in outs)
            return {"question": question, "answer": ans[:S.MAX_ANSWER_CHARS], "raw": " / ".join(o["raw"] for o in outs),
                    "qtype": "split", "parts": len(subs), "sources": [s for o in outs for s in o["sources"]],
                    "context": "\n".join(o["context"] for o in outs)}
        p = plan(question)
        qtype = p["qtype"]
        if hits is None:
            hits = self.retriever.search(question, k=S.top_k_for(p))
        ctx = build_context(question, hits, qtype)
        try:
            raw = llm.chat(build_messages(question, hits, qtype, p["max_sent"], ctx=ctx), p["max_new"])
            post_type, post_sent, calcs = qtype, p["max_sent"], []
            if qtype == "calc":
                calcs = run_calc_all(raw, p["n_calc"])
                multi = p["parts"] >= 2
                if calcs:
                    post_type = "explain" if multi else "fact"
                    post_sent = max(2, p["parts"]) if multi else 3
                    raw = llm.chat(build_calc_answer_messages(question, hits, calcs, multi, post_sent),
                                   64 + 40 * p["parts"] if multi else S.MAX_NEW_TOKENS["fact"])
                else:                                   # 式が作れなければ事実として答え直す
                    post_type = "fact"
                    raw = llm.chat(build_messages(question, hits, "fact"), S.MAX_NEW_TOKENS["fact"])
            ans = postprocess(raw, post_type, post_sent)
            if calcs:
                ans = ensure_calc_values(ans, calcs)
            elif S.REPAIR and S.remaining() > 60:          # 答えの形を点検し、だめなときだけ1回直させる
                words = unit_missing(question, ans)
                if words:
                    msgs = build_messages(question, hits, post_type, post_sent, ctx=ctx)
                    msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": repair_message(question, words)}]
                    raw2 = llm.chat(msgs, p["max_new"])
                    ans2 = postprocess(raw2, "fact", 3)
                    if ans2 and not unit_missing(question, ans2):
                        ans, raw = ans2, raw + " ||直し|| " + raw2
        except Exception as e:
            self.log(f"生成に失敗: {type(e).__name__}: {str(e)[:120]}")
            raw, ans = "", ""
        if not ans.strip():
            ans = extractive(ctx)
        return {"question": question, "answer": ans[:S.MAX_ANSWER_CHARS], "raw": raw, "qtype": qtype,
                "parts": p["parts"], "sources": [c.source for c, _ in hits], "context": ctx}