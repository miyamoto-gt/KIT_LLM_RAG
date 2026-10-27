"""⑤〜⑧ を束ねる中心部品。answer(question) -> dict が唯一の入口。

自作の質問でのテストも、採点APIでの本番も、この関数を呼ぶだけ。
"""
import re
import sys

from KIT_LLM_RAG.config import CFG
from KIT_LLM_RAG.generator import Generator
from KIT_LLM_RAG.index import Retriever
from prompts import build_calc_answer_prompt, build_prompt
from router import classify
from tools import run_calc_tags

STOP_MARKERS = ("\n質問", "\n###", "\n資料", "\n例）", "\n\n")


def postprocess(text: str, qtype: str) -> str:
    for m in STOP_MARKERS:                       # 次の質問を書き始める等の暴走を切る
        if m in text:
            text = text.split(m)[0]
    text = text.strip()
    if qtype in ("fact", "calc"):
        text = text.splitlines()[0] if text else text
        text = re.sub(r"^(回答|答え)[:：]\s*", "", text)
    return text.strip()


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generate = Generator()

    def retrieve(self, question: str) -> list:
        hits = self.retriever.search(question)
        if CFG.web_fallback and (not hits or hits[0][1] < CFG.low_score):
            from web_fallback import expand_from_hits
            known = {c.source for c in self.retriever.chunks}
            new = expand_from_hits(question, hits, known)
            if new:
                self.retriever.add(new)          # 取得したページは以降の質問でも使える
                hits = self.retriever.search(question)
        return hits

    def answer(self, question: str) -> dict:
        qtype = classify(question)
        hits = self.retrieve(question)
        raw = self.generate(build_prompt(question, hits, qtype), CFG.max_new_tokens[qtype])

        if qtype == "calc":
            calc = run_calc_tags(raw)
            if calc:
                expr, value = calc
                raw = self.generate(build_calc_answer_prompt(question, hits, expr, value),
                                    CFG.max_new_tokens["fact"])
            else:                                # 式が作れなければ事実質問として答え直す
                raw = self.generate(build_prompt(question, hits, "fact"), CFG.max_new_tokens["fact"])

        return {
            "question": question,
            "answer": postprocess(raw, qtype),
            "qtype": qtype,
            "sources": [c.source for c, _ in hits],
            "top_score": hits[0][1] if hits else None,
        }


if __name__ == "__main__":
    rag = RAGPipeline()
    for q in sys.argv[1:] or ["図書館は何号館にありますか？"]:
        print(rag.answer(q))
