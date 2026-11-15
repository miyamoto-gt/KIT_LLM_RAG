"""Private問題用のエントリポイント（サーバーがネットワーク遮断の環境で実行する）。

使い方（どれでも動く）:
    python main.py                                   # questions.csv → submission.csv
    python main.py 質問.csv 回答.csv
    python main.py --input 質問.csv --output 回答.csv   # -i/-o, --questions/--answers なども可
    環境変数 QUESTIONS_PATH / INPUT_PATH, OUTPUT_PATH / ANSWERS_PATH でも指定できる

方針（サーバーの環境が手元と違っても止まらない）:
  - 引数は知らないものがあっても無視する（argparse のエラーで終了コード2にならない）
  - 質問ファイルが見つからなければ、近くの CSV から question_id,question のものを探す
  - torch などの import やモデル・索引の読み込みに失敗しても、「不明」で埋めた回答CSVは必ず書く
  - エラーの内容は標準エラー出力と run_error.log に残し、最後は必ず終了コード0で終える
  - モデルの置き場所が決まっている場合は、環境変数 MODEL_PATH / EMBED_MODEL_PATH で指定できる
"""
import argparse
import csv
import glob
import os
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # ネットワーク遮断環境でダウンロードを試みて固まらない
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
START_DIR = os.getcwd()                                # サーバーが実行したときの作業フォルダ
sys.path.insert(0, HERE)

FALLBACK = "不明"
MAX_LEN = 2000
LOG = os.path.join(HERE, "run_error.log")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


# ---------- 入出力の場所 ----------
def parse_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("pos", nargs="*")
    ap.add_argument("--input", "-i", "--questions", "--question", "--in", "--input_path", "--input-path", dest="input")
    ap.add_argument("--output", "-o", "--out", "--answers", "--answer", "--output_path", "--output-path", dest="output")
    args, unknown = ap.parse_known_args()
    if unknown:
        log(f"知らない引数は無視します: {unknown}")
    inp = args.input or (args.pos[0] if len(args.pos) >= 1 else None) \
        or os.environ.get("QUESTIONS_PATH") or os.environ.get("INPUT_PATH")
    out = args.output or (args.pos[1] if len(args.pos) >= 2 else None) \
        or os.environ.get("OUTPUT_PATH") or os.environ.get("ANSWERS_PATH") or "submission.csv"
    return inp, out


def resolve(path: str) -> str:
    """相対パスは、サーバーの作業フォルダ → main.py のフォルダ の順に探す。"""
    if os.path.isabs(path):
        return path
    for base in (START_DIR, HERE):
        p = os.path.join(base, path)
        if os.path.exists(p):
            return p
    return os.path.join(START_DIR, path)


def looks_like_questions(path: str) -> bool:
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            head = f.readline().lower()
        return "question" in head and "answer" not in head
    except Exception:
        return False


def find_questions(given):
    if given:
        p = resolve(given)
        if os.path.exists(p):
            return p
        log(f"指定された質問ファイルが見つかりません: {given}")
    for name in ("questions.csv", "question.csv", "test.csv", "private.csv", "input.csv"):
        for base in (START_DIR, HERE, "/data", "/input", "/workspace"):
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    for base in (START_DIR, HERE, "/data", "/input", "/workspace"):   # 名前が違っても中身で探す
        for p in sorted(glob.glob(os.path.join(base, "**", "*.csv"), recursive=True))[:200]:
            if looks_like_questions(p):
                return p
    return None


def read_questions(path: str) -> list:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = list(rows[0].keys())
    id_col = next((c for c in cols if c and "id" in c.lower()), cols[0])
    q_col = next((c for c in cols if c and "question" in c.lower() and c != id_col), cols[-1])
    return [((r.get(id_col) or "").strip(), (r.get(q_col) or "").strip()) for r in rows]


def write_answers(path: str, answers: list) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "answer"])
        w.writerows(answers)


def clean(ans) -> str:
    ans = " ".join(str(ans or "").split())
    return (ans or FALLBACK)[:MAX_LEN]


# ---------- RAG ----------
_RAG = None
_RAG_TRIED = False                                         # 読み込みは1回だけ試す（失敗しても毎問やり直さない）


def load_rag():
    """モデルと索引を読み込む。失敗したら None（呼び出し側で「不明」にする）。"""
    global _RAG, _RAG_TRIED
    if _RAG_TRIED:
        return _RAG
    _RAG_TRIED = True
    cwd = os.getcwd()
    try:
        os.chdir(HERE)                                     # 索引などを相対パスで読む
        from config import CFG
        CFG.web_fallback = False
        if os.environ.get("MODEL_PATH"):
            CFG.llm_model = os.environ["MODEL_PATH"]
        if os.environ.get("EMBED_MODEL_PATH"):
            CFG.embed_model = os.environ["EMBED_MODEL_PATH"]
        from pipeline import RAGPipeline
        _RAG = RAGPipeline()
    except BaseException:                                  # ImportError や SystemExit も含めて止めない
        log("RAGPipeline の初期化に失敗しました:\n" + traceback.format_exc())
        _RAG = None
    finally:
        os.chdir(cwd)
    return _RAG


def answer(question: str) -> str:
    """質問1つに答える（関数として呼ばれる形式にも対応）。"""
    rag = load_rag()
    if rag is None:
        return FALLBACK
    cwd = os.getcwd()
    try:
        os.chdir(HERE)
        return clean(rag.answer(question)["answer"])
    except BaseException:
        log(f"回答中のエラー: {question[:40]}\n" + traceback.format_exc())
        return FALLBACK
    finally:
        os.chdir(cwd)


def main() -> None:
    inp, out = parse_args()
    out = out if os.path.isabs(out) else os.path.join(START_DIR, out)
    qpath = find_questions(inp)
    if not qpath:
        log("質問ファイルが見つかりませんでした。空の回答CSVを書いて終了します。")
        write_answers(out, [])
        return
    qs = read_questions(qpath)
    print(f"{qpath}: {len(qs)}問 → {out}", flush=True)

    answers, t0 = [], time.time()
    for i, (qid, q) in enumerate(qs, 1):
        answers.append((qid, answer(q)))
        print(f"[{i}/{len(qs)}] {qid} {time.time() - t0:.0f}s", flush=True)
        if i % 10 == 0:                                    # 途中で止められても、そこまでの回答は残す
            write_answers(out, answers + [(x, FALLBACK) for x, _ in qs[i:]])
    write_answers(out, answers)
    n_fb = sum(a == FALLBACK for _, a in answers)
    print(f"→ {out} に {len(answers)}問を書き出しました（「{FALLBACK}」{n_fb}問）", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        log("main で想定外のエラー:\n" + traceback.format_exc())
    sys.exit(0)