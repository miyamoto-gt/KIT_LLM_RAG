"""埋め込み API が「どのくらいの量なら通るか」を測る。python submit.py run ping2 で実行する。
1件・4件・8件・32件（それぞれ 500 字の資料）を順に送り、成功・失敗と秒数を表示する。"""
import csv, os, time
from openai import OpenAI

base = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
c = OpenAI(base_url=base, api_key=os.environ.get("OPENAI_API_KEY", "x"), timeout=120, max_retries=0)
model = os.environ.get("EMBEDDING_MODEL", "sarashina-embedding-v2-1b")
doc = "text: " + ("金沢工業大学は石川県野々市市にある私立大学で、1965年に開学した。" * 20)[:500]
for n in (1, 1, 4, 8, 32):
    t = time.time()
    try:
        r = c.embeddings.create(model=model, input=[doc] * n)
        print(f"{n:3d}件: OK {time.time() - t:.1f}秒（{len(r.data)}件・{len(r.data[0].embedding)}次元）", flush=True)
    except Exception as e:
        print(f"{n:3d}件: 失敗 {time.time() - t:.1f}秒 {type(e).__name__}: {str(e)[:90]}", flush=True)
    time.sleep(2)

qs = list(csv.DictReader(open(os.environ["INPUT_PATH"], encoding="utf-8-sig")))
with open(os.environ["OUTPUT_PATH"], "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["question_id", "answer"])
    w.writerows([[q["question_id"], "不明"] for q in qs])