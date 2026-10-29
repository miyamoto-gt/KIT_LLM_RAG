"""評価セットで全問を解き、キーワード一致で自動採点する（外部API不要）。

使い方:
  python evaluate.py                      # eval_set.json の全問
  python evaluate.py --category C_図書館フロア
  python evaluate.py --ids A01 C01 F03

判定:
  expect : [[A, B], [C]] のような「グループのリスト」。各グループのどれか1つが答えに含まれれば、そのグループは合格。
           すべてのグループが合格なら正解。expect が空の問題は採点せず、答えだけ表示する（正解を要確認の問題用）。
  forbid : 答えに含まれてはいけない言葉（よくある誤り）。1つでも含まれたら不正解。
  比較の前に、全角半角・空白・数字のカンマ・「円」をそろえる。
結果:
  最後に分野別と全体の正解率を表示する。
  eval_report.json      … 全問の答え・判定・根拠URL
  eval_failures.jsonl   … 間違えた問題だけ（1行1問。直したら --ids で解き直せる）
"""
import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict


def _kanji_amount(s: str) -> str:
    """「171万5千」「20万」「1万4850」「5千」のような書き方を数字（1715000 など）に直す。"""
    def man(m):
        a = int(m.group(1)) * 10000
        rest = m.group(2) or ""
        if rest.endswith("千"):
            a += int(rest[:-1]) * 1000
        elif rest:
            a += int(rest)
        return str(a)
    s = re.sub(r"(\d+)万(\d+千|\d{1,4}(?!\d))?", man, s)
    s = re.sub(r"(\d+)千", lambda m: str(int(m.group(1)) * 1000), s)
    return s


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for ch in (" ", "　", ",", "，", "円", "\n"):
        s = s.replace(ch, "")
    return _kanji_amount(s)


def judge(answer: str, expect: list, forbid: list):
    """(判定, 詳細) を返す。判定は 'pass' / 'fail' / 'skip'。"""
    a = norm(answer)
    hit_forbid = [w for w in forbid if norm(w) in a]
    if not expect:
        return "skip", "正解未設定"
    missing = [g for g in expect if not any(norm(w) in a for w in g)]
    if hit_forbid:
        return "fail", f"禁止語: {hit_forbid}"
    if missing:
        return "fail", f"不足: {[g[0] for g in missing]}"
    return "pass", ""


def free_memory() -> None:
    """MacのGPU（MPS）やCUDAのメモリを片付ける。長く回すと詰まって以降が全部失敗することがあるため。"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def ask(rag, question: str) -> dict:
    """1問解く。失敗したらメモリを片付けて1回だけやり直す。それでもだめなら空の答えとエラー内容を返す。"""
    for attempt in (1, 2):
        try:
            r = rag.answer(question)
            free_memory()
            return r
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            print(f"    !! エラー（{attempt}回目）: {err[:200]}", flush=True)
            free_memory()
    return {"answer": "", "sources": [], "qtype": "error", "error": err}


def main(path: str, category: str = None, ids: list = None) -> None:
    items = json.load(open(path, encoding="utf-8"))
    if category:
        items = [q for q in items if q["category"] == category]
    if ids:
        items = [q for q in items if q["id"] in ids]

    from pipeline import RAGPipeline
    rag = RAGPipeline()

    report, stats = [], defaultdict(lambda: [0, 0])
    for q in items:
        t = time.time()
        r = ask(rag, q["question"])                 # 1問のエラーで全体を止めない
        verdict, detail = judge(r["answer"], q.get("expect", []), q.get("forbid", []))
        if verdict != "skip":
            stats[q["category"]][1] += 1
            stats[q["category"]][0] += verdict == "pass"
        mark = {"pass": "○", "fail": "×", "skip": "－"}[verdict]
        print(f"{mark} {q['id']} [{r.get('qtype')}] {q['question']}\n    → {r['answer']}"
              + (f"\n    ({detail})" if detail else ""), flush=True)
        report.append({**q, "answer": r["answer"], "verdict": verdict, "detail": detail,
                       "qtype": r.get("qtype"), "sources": r.get("sources", []), "error": r.get("error"),
                       "sec": round(time.time() - t, 1)})

    print("\n===== 分野別の正解率 =====")
    total_ok = total_n = 0
    for cat in sorted(stats):
        ok, n = stats[cat]
        total_ok += ok
        total_n += n
        print(f"{cat:20s} {ok:3d} / {n:3d}  ({ok / n:.0%})")
    json.dump(report, open("eval_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    failures = [r for r in report if r["verdict"] == "fail"]
    with open("eval_failures.jsonl", "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps({k: r.get(k) for k in ("id", "category", "question", "answer", "detail",
                                                       "expect", "forbid", "qtype", "sources", "error")},
                               ensure_ascii=False) + "\n")
    print("=" * 40)
    if total_n:
        print(f"正解率: {total_ok} / {total_n} = {total_ok / total_n:.1%}")
    n_err = sum(r["qtype"] == "error" for r in report)
    print(f"間違えた問題: {len(failures)} 問 → eval_failures.jsonl（うちエラー {n_err} 問）")
    print(f"全問の詳細: eval_report.json")
    if failures:
        print("解き直す: python evaluate.py --ids " + " ".join(r["id"] for r in failures))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="eval_set.json")
    ap.add_argument("--category")
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    main(a.set, a.category, a.ids)