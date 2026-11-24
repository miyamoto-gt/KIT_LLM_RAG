"""正解の根拠を含むチャンクが、検索のどこにいるかを調べる（埋め込みは質問1回ずつだけ API を使う）。

  python evaluation/rank_check.py evaluation/sets/my_new69.json q064 q074
  python evaluation/rank_check.py evaluation/sets/public_set.json q088

・根拠（evidence）の語を含むチャンクを探し、埋め込み・BM25・最終の並びで何位かを出す
・最終の上位 top_k（生成に渡すもの）より上の資料も出す（何に負けているか）
・資料の埋め込みはキャッシュ（~/.cache/kitrag）から読む。無ければ作り直すので時間と API を使う
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))
cfg = json.loads((Path.home() / ".kitrag.json").read_text(encoding="utf-8"))
url = cfg["url"].rstrip("/") + "/api/llm/v1"
os.environ.update({"LLM_BASE_URL": url, "OPENAI_BASE_URL": url, "OPENAI_API_KEY": cfg["token"],
                   "KITRAG_TIME_LIMIT": str(10 ** 9)})

from kitrag import settings as S  # noqa: E402
from kitrag.knowledge import load_chunks  # noqa: E402
from kitrag.retriever import Retriever  # noqa: E402


def nz(t):
    return re.sub(r"\s+", "", t)


def main():
    qs = {q["id"]: q for q in json.load(open(sys.argv[1], encoding="utf-8"))}
    ids = sys.argv[2:] or list(qs)
    chunks = load_chunks()
    ret = Retriever(chunks, log=lambda *a: None)
    for qid in ids:
        q = qs[qid]
        ev = [w for g in q.get("evidence") or q.get("expect") for w in g]
        gold = [i for i, c in enumerate(chunks) if any(nz(w) in nz(c.text) for w in ev)]
        print(f"\n=== {qid} {q['question']}")
        print(f"根拠 {ev} を含むチャンク {len(gold)} 件")
        out, dbg = ret.search(q["question"], k=30, debug=True)
        final = [c.id for c, _ in out]
        for i in gold[:8]:
            c = chunks[i]
            d = [rl.index(i) + 1 if i in rl else "-" for rl in dbg["dense_each"]]
            b = dbg["bm25"].index(i) + 1 if i in dbg["bm25"] else "-"
            f = final.index(c.id) + 1 if c.id in final else "-"
            print(f"  #{i} 埋め込み{d} BM25 {b} 最終 {f}  {c.source[:60]}")
            print(f"      {nz(c.text)[:90]}")
        print(f"  最終の上位{S.TOP_K}件:")
        for c, _ in out[:S.TOP_K]:
            print(f"    {c.source[:60]}  {nz(c.text)[:60]}")


if __name__ == "__main__":
    main()