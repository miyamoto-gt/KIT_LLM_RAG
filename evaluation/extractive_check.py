"""簡易回答（締め切り後・生成失敗時の答え）の正解率を、API を使わずに測る。

  python evaluation/extractive_check.py evaluation/sets/my_new69.json evaluation/sets/public_set.json evaluation/sets/paraphrase_set.json

・本番の quick() と同じ流れ：BM25 だけで検索 → build_context → extractive
・「前（最初の文）」と「新（質問と重なる文）」を、評価セットのキーワードで比べる
・埋め込みも生成も呼ばない（ネット不要・数十秒）
・-v を付けると、外した問題の候補の上位3文と点数も出す（直す手がかり）
"""
import json
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "submission"))
os.environ.setdefault("KITRAG_TIME_LIMIT", str(10 ** 9))

from kitrag import settings as S  # noqa: E402
S.EMBED_MODELS = []                  # 埋め込みは使わない（API を呼ばない）
from kitrag.knowledge import load_chunks  # noqa: E402
from kitrag.pipeline import extractive, rank_sentences  # noqa: E402
from kitrag.prompts import build_context  # noqa: E402
from kitrag.retriever import Retriever  # noqa: E402
from kitrag.router import plan  # noqa: E402


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    for c in " 　,，円\n":
        s = s.replace(c, "")
    return s


def hit(a, expect):
    return all(any(norm(w) in norm(a) for w in g) for g in expect)


def main():
    ret = Retriever(load_chunks(), log=lambda *a: None)
    verbose = "-v" in sys.argv
    for path in [a for a in sys.argv[1:] if a != "-v"]:
        qs = [q for q in json.load(open(path, encoding="utf-8")) if q.get("expect")]
        old = new = inctx = 0
        for q in qs:
            ctx = build_context(q["question"], ret.search(q["question"], use_dense=False), plan(q["question"])["qtype"])
            a0, a1 = extractive(ctx), extractive(ctx, q["question"])
            c = hit(ctx, q["expect"])
            inctx += c
            old += hit(a0, q["expect"])
            new += hit(a1, q["expect"])
            if c and not hit(a1, q["expect"]):
                print(f"  × {q['id']} {q['question'][:28]} → {a1[:60]}")
                if verbose:
                    for sc, s, has in rank_sentences(ctx, q["question"])[:3]:
                        print(f"        {sc:5.1f} {'形○' if has else '  '} {s[:70]}")
        n = len(qs)
        print(f"{os.path.basename(path)}: 資料に答えあり {inctx}/{n}｜前 {old}/{n}｜新 {new}/{n}\n")


if __name__ == "__main__":
    main()