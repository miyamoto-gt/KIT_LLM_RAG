"""Private問題用のエントリポイント（サーバーがネットワーク遮断の環境で実行する）。

使い方:
    python main.py                               # questions.csv → submission.csv
    python main.py 質問.csv --out 回答.csv

方針:
  - ネットワークに出ない（Web補完を無効化、Hugging Face もオフライン指定）
  - 1問でエラーが出ても止めずに「不明」で埋め、最後まで回答CSVを書き出す
  - 出力は question_id,answer の2列（UTF-8）
"""
import argparse
import csv
import os
import sys
import time

# ネットワーク遮断環境でダウンロードを試みて固まらないようにする
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)                     # index や資料を相対パスで読めるようにする

from config import CFG             # noqa: E402

try:
    CFG.web_fallback = False       # オフラインなのでWeb補完は使わない
except Exception:
    pass

from pipeline import RAGPipeline   # noqa: E402

FALLBACK = "不明"


def read_questions(path: str) -> list:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = list(rows[0].keys())
    id_col = next((c for c in cols if "id" in c.lower()), cols[0])
    q_col = next((c for c in cols if "question" in c.lower() and c != id_col), cols[-1])
    return [(r[id_col].strip(), r[q_col].strip()) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="?", default="questions.csv")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    qs = read_questions(a.questions)
    print(f"{a.questions}: {len(qs)}問")

    try:
        rag = RAGPipeline()
    except Exception as e:             # モデルや索引が読めない場合も、空欄のCSVは必ず出す
        print("RAGPipeline の初期化に失敗:", repr(e))
        rag = None

    answers = []
    t0 = time.time()
    for i, (qid, q) in enumerate(qs, 1):
        ans = FALLBACK
        if rag is not None:
            try:
                ans = rag.answer(q)["answer"].strip() or FALLBACK
            except Exception as e:
                print(f"[{qid}] エラー: {e!r}")
        answers.append((qid, ans.replace("\n", " ")))
        print(f"[{i}/{len(qs)}] {qid} {time.time() - t0:.0f}s")

    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "answer"])
        w.writerows(answers)
    n_fb = sum(a_ == FALLBACK for _, a_ in answers)
    print(f"→ {a.out} に {len(answers)}問を書き出しました（「{FALLBACK}」{n_fb}問）")


if __name__ == "__main__":
    main()