"""埋め込み API が 502 を返す原因になっているチャンクを探す。

  python build/find_bad_chunks.py                 # API を使わずに、怪しい文字・長さのチャンクを一覧にする
  python build/find_bad_chunks.py --api           # 32件ずつ順番に送り、失敗したまとまりだけ二分して原因のチャンクを特定する

--api の使用回数の目安：約280回（32件 × 約280）＋ 失敗1か所につき約10回。
失敗したら10秒待ってから次へ進む（サーバーが戻るのを待つ）。結果は outputs/bad_chunks.txt にも書く。
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "submission"))


def setup_api():
    cfg = json.loads((Path.home() / ".kitrag.json").read_text(encoding="utf-8"))
    url = cfg["url"].rstrip("/") + "/api/llm/v1"
    os.environ.update({"LLM_BASE_URL": url, "OPENAI_BASE_URL": url, "OPENAI_API_KEY": cfg["token"],
                       "KITRAG_TIME_LIMIT": str(10 ** 9)})


def odd_chars(t):
    return [c for c in t if unicodedata.category(c) in ("Cc", "Cf", "Cs", "Co", "Cn") and c not in "\n\t"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    if a.api:
        setup_api()
    from kitrag import settings as S
    from kitrag.knowledge import load_chunks
    chunks = load_chunks()
    print(f"チャンク {len(chunks)} 件")

    # ---- API を使わない点検 ----
    sus = []
    for i, c in enumerate(chunks):
        bad = odd_chars(c.text)
        longest = max((len(w) for w in c.text.split()), default=0)      # 空白なしで続く長い塊（URL・記号の羅列など）
        if bad or len(c.text) > 1500 or longest > 400:
            sus.append((i, c, bad, longest))
    print(f"\n【点検】怪しいチャンク {len(sus)} 件（変な文字・1500字超・空白なしの長い塊）")
    for i, c, bad, longest in sus[:30]:
        print(f"  #{i} {len(c.text)}字 塊{longest} 変な文字{len(bad)} {repr(''.join(bad[:3]))} {c.source[:70]}")
    if not a.api:
        return

    # ---- API で特定 ----
    from kitrag import llm
    texts = [S.DOC_PREFIX + c.text for c in chunks]
    bad_ids, calls = [], [0]

    def ok(lo, hi):
        calls[0] += 1
        try:
            llm.embed(texts[lo:hi], tries=1)
            return True
        except Exception as e:
            code = getattr(e, "status_code", None)
            if code == 429:
                print("回数制限（429）になったので止めます")
                raise SystemExit(1)
            print(f"  失敗 {lo}〜{hi - 1}（{hi - lo}件）: {type(e).__name__} {code or ''}", flush=True)
            time.sleep(10)
            return False

    def bisect(lo, hi):
        if hi - lo == 1:
            bad_ids.append(lo)
            return
        mid = (lo + hi) // 2
        for x, y in ((lo, mid), (mid, hi)):
            if not ok(x, y):
                bisect(x, y)

    t0 = time.time()
    for lo in range(0, len(texts), a.batch):
        hi = min(lo + a.batch, len(texts))
        if not ok(lo, hi):
            if ok(lo, hi):                       # もう一度送って通るなら、一時的な不調
                continue
            bisect(lo, hi)
        if (lo // a.batch) % 20 == 0:
            print(f"  {hi}/{len(texts)}（{time.time() - t0:.0f}秒・{calls[0]}回）", flush=True)
    print(f"\n【結果】502 の原因と思われるチャンク {len(bad_ids)} 件（API {calls[0]}回）")
    lines = []
    for i in bad_ids:
        c = chunks[i]
        lines.append(f"#{i} {c.source}\n  {c.text[:200]!r}")
    print("\n".join(lines) or "  なし（一時的な不調だった）")
    os.makedirs(ROOT / "outputs", exist_ok=True)
    (ROOT / "outputs" / "bad_chunks.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
    