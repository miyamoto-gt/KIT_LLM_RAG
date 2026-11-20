"""提出用エントリポイント（運営テンプレートの約束に合わせた版）。

サーバーはこのフォルダを zip で受け取り、ネットワークから切り離したサンドボックスで `python main.py` を実行する。
  INPUT_PATH       質問CSV（question_id,question）。本番では Private 問題だけが入る
  OUTPUT_PATH      回答CSV（question_id,answer）をここに書く
  LLM_BASE_URL     OpenAI 互換 API。LLM_MODEL（sarashina2.2-3b）。埋め込みモデルは kitrag/settings.py で決める

止まらないための約束：
  ・起動直後に、全問「不明」の回答CSVを先に書いておく（途中で止められても空にならない）
  ・制限時間（30分）の手前（TIME_LIMIT）で打ち切る。答えが出ていない質問は資料の一文で答えて書き出し、
    応答待ちの API 呼び出しは待たずに終了する（os._exit）
  ・1問の失敗で全体を止めない。最後は必ず終了コード0
"""
import csv
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kitrag import settings as S  # noqa: E402

START = S.START

FALLBACK = S.FALLBACK


def log(msg):
    print(f"[{time.time() - START:6.0f}s] {msg}", flush=True)


def read_questions(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    cols = list(rows[0].keys())
    id_col = "question_id" if "question_id" in cols else next((c for c in cols if c and "id" in c.lower()), cols[0])
    q_col = "question" if "question" in cols else cols[-1]
    return [((r.get(id_col) or "").strip(), (r.get(q_col) or "").strip()) for r in rows]


def write_answers(path, answers):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["question_id", "answer"])
        for qid, a in answers:
            w.writerow([qid, " ".join(str(a or FALLBACK).split())[:S.MAX_ANSWER_CHARS] or FALLBACK])
    os.replace(tmp, path)


def main():
    in_path = os.environ.get("INPUT_PATH", os.path.join(HERE, "questions.csv"))
    out_path = os.environ.get("OUTPUT_PATH", os.path.join(HERE, "submission.csv"))
    qs = read_questions(in_path)
    answers = {qid: FALLBACK for qid, _ in qs}
    write_answers(out_path, [(qid, answers[qid]) for qid, _ in qs])          # まず空にならない CSV を置く
    log(f"質問 {len(qs)} 問")

    try:
        from kitrag.knowledge import load_chunks
        from kitrag.pipeline import RAG
        chunks = load_chunks()
        log(f"資料 {len(chunks)} チャンク")
        rag = RAG(chunks, log=log)
    except Exception:
        log("初期化に失敗:\n" + traceback.format_exc())
        return

    lock = threading.Lock()
    done = [0]

    def quick(q):
        """締め切り後の答え：BM25 だけで検索し、資料の一文で答える（API は呼ばない）。"""
        try:
            from kitrag.pipeline import extractive
            from kitrag.prompts import build_context
            from kitrag.router import plan
            hits = rag.retriever.search(q, use_dense=False)
            return extractive(build_context(q, hits, plan(q)["qtype"]))
        except Exception:
            return FALLBACK

    def solve(item):
        qid, q = item
        if S.remaining() < 20:
            a = quick(q)
        else:
            try:
                a = rag.answer(q)["answer"]
            except Exception as e:
                log(f"{qid}: {type(e).__name__}")
                a = FALLBACK
        with lock:
            answers[qid] = a or FALLBACK
            done[0] += 1
            if done[0] % 10 == 0:
                write_answers(out_path, [(x, answers[x]) for x, _ in qs])
                log(f"{done[0]}/{len(qs)} 問")
        return qid

    pool = ThreadPoolExecutor(max_workers=S.WORKERS)
    futs = {pool.submit(solve, item): item[0] for item in qs}
    finished, pending = wait(futs, timeout=max(S.remaining(), 0))
    pool.shutdown(wait=False, cancel_futures=True)
    late = [futs[f] for f in pending]
    if late:
        log(f"締め切りのため {len(late)} 問を資料の一文で答えます")
        qmap = dict(qs)
        with lock:
            for qid in late:
                answers[qid] = quick(qmap[qid]) or FALLBACK
    with lock:
        write_answers(out_path, [(qid, answers[qid]) for qid, _ in qs])
    n_fb = sum(answers[q] == FALLBACK for q, _ in qs)
    try:
        from kitrag.llm import STATS
        if STATS:
            st = sorted(STATS)
            log(f"生成 {len(st)} 回：中央値 {st[len(st)//2]:.0f}秒・最大 {st[-1]:.0f}秒（並列 {S.WORKERS}）")
    except Exception:
        pass
    log(f"{len(qs)} 問の回答を書き出しました（「{FALLBACK}」{n_fb} 問・締め切りで簡易回答 {len(late)} 問）→ {out_path}")

if __name__ == "__main__":
    try:
        main()
    except BaseException:
        log("想定外のエラー:\n" + traceback.format_exc())
    sys.stdout.flush()
    os._exit(0)                     # 応答待ちのスレッドを待たずに終える
