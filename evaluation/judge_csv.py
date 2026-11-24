"""回答CSV を評価セットのキーワードで採点する（API は使わない）。

  python evaluation/judge_csv.py q69_w8.csv evaluation/sets/my_new69.json
  python evaluation/judge_csv.py submission.csv evaluation/sets/public_set.json

・正解・不正解の数と、間違えた問題を表示する
・資料の切れ端のような答え（簡易回答らしいもの）も数える
"""
import csv
import json
import re
import sys
import unicodedata


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    for c in " 　,，円\n":
        s = s.replace(c, "")
    return s


def looks_extractive(a):
    """簡易回答（資料の一文をそのまま返したもの）らしい答え。"""
    return bool(re.match(r"^[（(]?TOP\s*>|^[>＞]|^[-・■●#|]", a or "")) or not re.search(r"(です|ます|でした|ません)[。）)]?$", (a or "").strip())


def main():
    ans = {r["question_id"]: r["answer"] for r in csv.DictReader(open(sys.argv[1], encoding="utf-8"))}
    qs = json.load(open(sys.argv[2], encoding="utf-8"))
    ok = ng = 0
    ext = []
    for q in qs:
        if q["id"] not in ans:
            continue
        a = ans[q["id"]]
        if looks_extractive(a):
            ext.append(q["id"])
        if not q.get("expect"):
            continue
        hit = all(any(norm(w) in norm(a) for w in g) for g in q["expect"])
        ok += hit
        ng += not hit
        if not hit:
            print(f"× {q['id']} {q['question'][:30]} → {a[:60]}")
    print(f"\n正解 {ok}/{ok + ng}（{100 * ok / max(ok + ng, 1):.0f}%）")
    print(f"簡易回答らしい答え {len(ext)} 問: {' '.join(ext)}")


if __name__ == "__main__":
    main()
