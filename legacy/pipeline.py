"""⑤〜⑧ を束ねる中心部品。answer(question) -> dict が唯一の入口。

自作の質問でのテストも、採点APIでの本番も、この関数を呼ぶだけ。
"""
import re
import sys

from config import CFG
from generator import Generator
from index import Retriever
from prompts import build_calc_answer_prompt, build_prompt
from router import plan
from tools import run_calc_all
from postfix import cut_english

STOP_MARKERS = ("\n質問", "\n###", "\n資料", "\n例）")
LIST_ITEM = re.compile(r"\n\s*(?:[-・*●◆]|\d+\s*[.．)）、])\s*")
# 「以下の通りです。」のような前置きだけの行（中身は次の行から続く）
PREAMBLE = re.compile(r"(以下の(通り|とおり|よう)|次の(通り|とおり|よう)|以下になります|以下です)")
LABEL_ITEM = re.compile(r"^[^。]{1,40}?[：:]\s*\S")     # 「1966年：経営工学科」のような見出し付きの行


def drop_preamble(text: str) -> str:
    """1行目が前置き（以下の通りです 等）で、その後に中身があれば、前置きの行を外す。
    事実型は1行目だけを答えにするので、前置きが残ると中身が全部消えてしまう。"""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= 2 and len(lines[0]) <= 80 and PREAMBLE.search(lines[0]):
        return "\n".join(lines[1:])
    return text


def is_item(line: str) -> bool:
    return bool(LIST_ITEM.match("\n" + line)) or (bool(LABEL_ITEM.match(line)) and "。" not in line[:-1])


def strip_item(line: str) -> str:
    m = LIST_ITEM.match("\n" + line)
    return line[m.end() - 1:].strip() if m else line.strip()


def ensure_calc_values(answer: str, calcs: list) -> str:
    """計算ツールで出した値が答えに書かれていなければ、末尾に（式 = 値）を添える。
    モデルが値を書き写し忘れても、計算結果は必ず答えに残る。"""
    def digits(s):
        return re.sub(r"[,，\s円人名]", "", str(s))
    missing = [(e, v) for e, v in calcs if digits(v) not in digits(answer)]
    if not missing:
        return answer
    add = "、".join(f"{e} = {v}" for e, v in missing)
    return f"{answer}（計算：{add}）" if answer else f"{add}"


def calc_final(question: str, hits: list, calcs: list, p: dict):
    """計算結果を添えて最終回答を書かせる設定。(プロンプト, 出力上限, 後処理の種類, 文数) を返す。
    答えの部分が1つなら今までどおり1行。複数なら説明型で複数文（調べる値も答えに残す）。"""
    multi = p["parts"] >= 2
    prompt = build_calc_answer_prompt(question, hits, calcs, multi=multi)
    if multi:
        return prompt, 64 + 40 * p["parts"], "explain", max(2, p["parts"])
    return prompt, CFG.max_new_tokens["fact"] + 32 * (len(calcs) - 1), "fact", 3


def postprocess(text: str, qtype: str, max_sent: int = 3) -> str:
    text = text.strip()                          # 先頭の改行を先に取り除く
    for m in STOP_MARKERS:                       # 次の質問を書き始める等の暴走を切る
        if m in text:
            text = text.split(m)[0]
    text = cut_english(text.strip())             # 途中から英語に切り替わったら、その手前の「。」で切る
    after = drop_preamble(text)                  # 「以下の通りです。」だけの1行目を外す
    listed = after != text                       # 前置きの後ろは一覧（記号が無くても項目とみなす）
    text = after
    if qtype in ("fact", "calc"):
        lines = [l.strip() for l in text.split("\n\n")[0].splitlines() if l.strip()]
        if len(lines) >= 2 and listed and all(len(l) <= 60 for l in lines):
            text = "、".join(strip_item(l).rstrip("。") for l in lines) + "。"
        elif len(lines) >= 2 and is_item(lines[0]):   # 一覧で答えている → 続く項目を「、」でつなぐ
            items = []
            for l in lines:
                if not is_item(l):
                    break
                items.append(strip_item(l).rstrip("。"))
            text = "、".join(items) + "。"
        else:
            text = lines[0] if lines else ""
        text = re.sub(r"^(回答|答え)[:：]\s*", "", text)
    if qtype == "explain":
        text = text.split("\n\n資料")[0]
        parts, items = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if is_item(line):                    # 箇条書き・「見出し：中身」の1行
                items.append(strip_item(line).rstrip("。"))
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
        if sentences:                            # 説明は最初の max_sent 文まで（部分が多い質問ほど長く）
            text = "".join(sentences[:max_sent])
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
        p = plan(question)                       # 種類・部分の数・出力の上限
        qtype = p["qtype"]
        hits = self.retrieve(question)
        raw = self.generate(build_prompt(question, hits, qtype), p["max_new"])

        calcs = []
        if qtype == "calc":
            calcs = run_calc_all(raw, p["n_calc"])
            if calcs:
                prompt, max_new, post_type, post_sent = calc_final(question, hits, calcs, p)
                raw = self.generate(prompt, max_new)
            else:                                # 式が作れなければ事実質問として答え直す
                raw = self.generate(build_prompt(question, hits, "fact"), CFG.max_new_tokens["fact"])

        if qtype == "calc" and calcs:
            answer = ensure_calc_values(postprocess(raw, post_type, post_sent), calcs)
        else:
            answer = postprocess(raw, qtype, p["max_sent"])
        return {
            "question": question,
            "answer": answer,
            "raw": raw, 
            "qtype": qtype,
            "parts": p["parts"],
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