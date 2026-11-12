"""⑤〜⑧ を束ねる中心部品。answer(question) -> dict が唯一の入口。

自作の質問でのテストも、採点APIでの本番も、この関数を呼ぶだけ。
"""
import re
import sys

from config import CFG
from generator import Generator
from index import Retriever
from prompts import build_calc_answer_prompt, build_prompt
from router import classify
from tools import run_calc_tags
from postfix import cut_english

STOP_MARKERS = ("\n質問", "\n###", "\n資料", "\n例）")
LIST_ITEM = re.compile(r"\n\s*(?:[-・*●◆]|\d+\s*[.．)）、])\s*")


def postprocess(text: str, qtype: str) -> str:
    text = text.strip()                          # 先頭の改行を先に取り除く
    for m in STOP_MARKERS:                       # 次の質問を書き始める等の暴走を切る
        if m in text:
            text = text.split(m)[0]
    text = cut_english(text.strip())             # 途中から英語に切り替わったら、その手前の「。」で切る
    if qtype in ("fact", "calc"):
        text = text.split("\n\n")[0]
        lines = [l for l in text.splitlines() if l.strip()]
        text = lines[0] if lines else ""
        text = re.sub(r"^(回答|答え)[:：]\s*", "", text)
    if qtype == "explain":
        text = text.split("\n\n資料")[0]
        parts, items = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = LIST_ITEM.match("\n" + line)
            if m:                                # 箇条書きの1行
                items.append(line[m.end() - 1:].strip())
                continue
            if items:                            # 箇条書きが終わったら「、」でつないで1文にする
                parts.append("、".join(items) + "。")
                items = []
            parts.append(line)
        if items:
            parts.append("、".join(items) + "。")
        text = "".join(parts).replace("。。", "。")
        if text and not text.endswith("。"):
            text += "。"
        sentences = re.findall(r"[^。]+。", text)
        if sentences:                            # 説明は最初の3文まで
            text = "".join(sentences[:3])
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
            "raw": raw, 
            "qtype": qtype,
            "sources": [c.source for c, _ in hits],
            "top_score": hits[0][1] if hits else None,
        }


def show(r: dict, show_raw: bool = False) -> None:
    """結果を見やすく表示する。"""
    print("\n" + "─" * 60)
    print(f"質問　：{r['question']}")
    print(f"回答　：{r['answer'] or '（回答なし）'}")
    score = f"{r['top_score']:.3f}" if r.get("top_score") is not None else "-"
    print(f"種類　：{r['qtype']}　　類似度：{score}")
    print("根拠　：")
    seen = []
    for s in r.get("sources", []):
        if s not in seen:
            seen.append(s)
            print(f"  {len(seen)}. {s}")
    if show_raw:
        print("生出力：")
        print("  " + r.get("raw", "").strip().replace("\n", "\n  ")[:500])
    print("─" * 60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="質問に答える（質問を省略すると対話モード）")
    ap.add_argument("questions", nargs="*", help="質問文（複数可）")
    ap.add_argument("--raw", action="store_true", help="モデルの生の出力も表示する")
    a = ap.parse_args()

    rag = RAGPipeline()
    if a.questions:
        for q in a.questions:
            show(rag.answer(q), a.raw)
    else:                                        # 対話モード：空行か Ctrl+C で終了
        print("質問を入力してください（空行で終了）")
        try:
            while True:
                q = input("\n> ").strip()
                if not q:
                    break
                show(rag.answer(q), a.raw)
        except (KeyboardInterrupt, EOFError):
            pass