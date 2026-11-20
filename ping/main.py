"""運営 API の混み具合を測る（生成1回・埋め込み1回だけ）。python submit.py run ping で実行する。"""
import csv, os, time
from openai import OpenAI

base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
c = OpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "x"), timeout=300, max_retries=0)
for name, fn in [
    ("埋め込み", lambda: c.embeddings.create(model=os.environ.get("EMBEDDING_MODEL", "sarashina-embedding-v2-1b"),
                                          input=["query: テスト"])),
    ("生成", lambda: c.chat.completions.create(model=os.environ.get("LLM_MODEL", "sarashina2.2-3b"),
                                             messages=[{"role": "user", "content": "こんにちは"}], max_tokens=8)),
]:
    t = time.time()
    try:
        fn()
        print(f"{name}: OK {time.time() - t:.1f}秒", flush=True)
    except Exception as e:
        print(f"{name}: 失敗 {time.time() - t:.1f}秒 {type(e).__name__}: {str(e)[:100]}", flush=True)

qs = list(csv.DictReader(open(os.environ["INPUT_PATH"], encoding="utf-8-sig")))
with open(os.environ["OUTPUT_PATH"], "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["question_id", "answer"])
    w.writerows([[q["question_id"], "不明"] for q in qs])
