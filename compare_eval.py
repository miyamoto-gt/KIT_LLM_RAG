"""evaluate.py の出力2つ（before / after）を比べて、正誤が変わった問題だけを表示する。

使い方:
    python compare_eval.py eval_before.txt eval_after.txt
"""
import re
import sys

LINE = re.compile(r"^\s*(\S)\s+(\S+)\s+\[(\w+)\]\s+(.*)$")   # 「○ A01 [fact] 質問」
ANS = re.compile(r"^\s*→\s*(.*)$")


def load(path):
    res, last = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                mark, qid, qtype, q = m.groups()
                res[qid] = {"ok": mark == "○", "mark": mark, "type": qtype, "q": q, "a": ""}
                last = qid
                continue
            a = ANS.match(line)
            if a and last:
                res[last]["a"] = a.group(1)
    return res


def summary(name, res):
    ok = sum(r["ok"] for r in res.values())
    by = {}
    for r in res.values():
        t = by.setdefault(r["type"], [0, 0])
        t[0] += r["ok"]
        t[1] += 1
    detail = "  ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(by.items()))
    print(f"{name:7s} {ok}/{len(res)}  ({detail})")


before, after = load(sys.argv[1]), load(sys.argv[2])
before = {k: v for k, v in before.items() if k in after}   # after で回した問題だけを比べる
summary("before", before)
summary("after", after)

worse = [k for k in before if k in after and before[k]["ok"] and not after[k]["ok"]]
better = [k for k in before if k in after and not before[k]["ok"] and after[k]["ok"]]

for title, ids in (("▼ 悪化（○→×）", worse), ("▲ 改善（×→○）", better)):
    print(f"\n{title}: {len(ids)}問")
    for k in ids:
        print(f"  {k} [{before[k]['type']}] {before[k]['q']}")
        print(f"     before: {before[k]['a']}")
        print(f"     after : {after[k]['a']}")

missing = set(after) - set(before)
if missing:
    print(f"\n片方にしかない問題: {sorted(missing)}")