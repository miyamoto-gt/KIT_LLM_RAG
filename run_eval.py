"""⑤⑧ 採点APIとのやりとり。全問をループして回答JSONを作る。

※ 採点APIの形式が未確定なので、キー名とURLは TODO。形式が分かったらここだけ直す。
使い方:
  python run_eval.py questions.json            # ローカルのJSONで実行
  python run_eval.py https://.../questions     # APIから取得して実行
"""
import json
import sys
import time

import requests

from KIT_LLM_RAG.pipeline import RAGPipeline

QUESTION_KEY = "question"   # TODO: 採点JSONのキー名に合わせる
ID_KEY = "id"               # TODO
SUBMIT_URL = None           # TODO: 回答を送るAPIのURL


def load_questions(src: str) -> list:
    if src.startswith("http"):
        data = requests.get(src, timeout=30).json()
    else:
        data = json.load(open(src, encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) and "questions" in data else data


def main(src: str) -> None:
    rag = RAGPipeline()
    results, log = [], []
    for i, item in enumerate(load_questions(src)):
        t = time.time()
        r = rag.answer(item[QUESTION_KEY])
        results.append({ID_KEY: item.get(ID_KEY, i), "answer": r["answer"]})
        log.append({**r, "sec": round(time.time() - t, 2)})
        print(f"[{i}] ({r['qtype']}) {r['question']} -> {r['answer']}")

    json.dump(results, open("answers.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(log, open("answers_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved answers.json / answers_log.json")
    if SUBMIT_URL:
        print(requests.post(SUBMIT_URL, json=results, timeout=60).text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "questions.json")
