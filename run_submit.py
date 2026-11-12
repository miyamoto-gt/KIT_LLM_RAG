"""本番提出用：questions.csv の全問に答えて submission.csv（question_id,answer）を作る。

使い方:
  python submit.py questions                 # questions.csv を取得
  python run_submit.py                       # → submission.csv
  python run_submit.py questions.csv --limit 3   # 先頭3問だけ（速度テスト）
  python submit.py submit submission.csv --name "baseline"

・1問ごとに answers_ckpt.jsonl へ追記するので、途中で止まっても再実行で続きから。
・空欄は0点なので、失敗・空の答えは FALLBACK で埋める。
・改行はスペースにし、2000文字で切る。最後に形式チェックをする。
"""
import argparse
import csv
import json
import os
import time

ID_CANDIDATES = ("question_id", "id", "qid")
Q_CANDIDATES = ("question", "text", "query", "質問")
CKPT = "answers_ckpt.jsonl"
FALLBACK = "不明"
MAX_LEN = 2000


def pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    raise SystemExit(f"{what}の列が見つかりません。列名: {cols}  → ID_CANDIDATES / Q_CANDIDATES に追加してください")


def load_questions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:   # BOMがあっても読める
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} に質問がありません")
    cols = list(rows[0].keys())
    id_col, q_col = pick(cols, ID_CANDIDATES, "ID"), pick(cols, Q_CANDIDATES, "質問")
    print(f"{path}: {len(rows)}問（ID列={id_col} / 質問列={q_col}）")
    return [(r[id_col], r[q_col]) for r in rows]           # IDは文字列のまま（001→1 を防ぐ）


def clean(ans):
    ans = " ".join((ans or "").split())                    # 改行・連続空白 → 1つの空白
    return (ans or FALLBACK)[:MAX_LEN]


def load_ckpt():
    done = {}
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done[r["question_id"]] = r["answer"]
    return done


def write_csv(questions, done, out):
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")  # 1行目は question_id,answer そのまま
        w.writerow(["question_id", "answer"])
        for qid, _ in questions:
            w.writerow([qid, clean(done.get(qid))])


def check(questions, out):
    with open(out, encoding="utf-8", newline="") as f:
        first = f.readline().rstrip("\n")
        assert first == "question_id,answer", f"1行目が違う: {first}"
        f.seek(0)
        rows = list(csv.reader(f))
    assert rows[0] == ["question_id", "answer"], f"ヘッダーが違う: {rows[0]}"
    body = rows[1:]
    assert len(body) == len(questions), f"行数 {len(body)} ≠ 問題数 {len(questions)}"
    assert [r[0] for r in body] == [q for q, _ in questions], "question_id が一致しない"
    assert all(len(r) == 2 and r[1].strip() for r in body), "空欄の回答がある"
    assert max(len(r[1]) for r in body) <= MAX_LEN, "2000文字を超える回答がある"
    n_fb = sum(r[1] == FALLBACK for r in body)
    print(f"形式チェックOK: {len(body)}問（「{FALLBACK}」で埋めたもの {n_fb}問）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="questions.csv")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--limit", type=int, help="先頭N問だけ解く（速度テスト用）")
    ap.add_argument("--redo-fallback", action="store_true", help="「不明」になった問題だけ解き直す")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    done = load_ckpt()
    if args.redo_fallback:
        done = {k: v for k, v in done.items() if clean(v) != FALLBACK}
    todo = [(qid, q) for qid, q in questions if qid not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"回答済み {len(done)}問 / これから {len(todo)}問")

    if todo:
        from evaluate import ask                            # 失敗時にメモリを片付けて1回やり直す
        from pipeline import RAGPipeline
        rag = RAGPipeline()
        t0 = time.time()
        with open(CKPT, "a", encoding="utf-8") as f:
            for i, (qid, q) in enumerate(todo, 1):
                t = time.time()
                r = ask(rag, q)
                ans = clean(r.get("answer"))
                done[qid] = ans
                f.write(json.dumps({"question_id": qid, "answer": ans, "qtype": r.get("qtype"),
                                    "sources": r.get("sources", [])[:3],
                                    "sec": round(time.time() - t, 1)}, ensure_ascii=False) + "\n")
                f.flush()
                avg = (time.time() - t0) / i
                print(f"[{i}/{len(todo)}] {time.time()-t:.1f}s 残り約{avg*(len(todo)-i)/60:.0f}分",
                      flush=True)

    if args.limit and len(done) < len(questions):
        print(f"--limit のため {args.out} は書き出しません（全問終わったら引数なしで実行）")
        return
    write_csv(questions, done, args.out)
    check(questions, args.out)
    print(f"→ python submit.py submit {args.out} --name \"<名前>\"")


if __name__ == "__main__":
    main()