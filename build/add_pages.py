"""指定した URL のページを取ってきて knowledge/ に足す（すでにあるページは取り直して上書き）。

  python build/add_pages.py submission/knowledge --check                       # 下の既定のページが資料にあるかだけ見る
  python build/add_pages.py submission/knowledge                                # 既定のページを足す・取り直す
  python build/add_pages.py submission/knowledge --urls https://... https://...  # ページを指定

・既定のページ：大学の「数字で見る」「就職実績」など、数字がまとまった入口ページ（下の DEFAULT）
・書き換える前の knowledge は outputs/knowledge_before_add/ に控える
・本文の取り出し方は refetch_thin.py と同じ（表は「| 区切り」、<dl> は「見出し：中身」）。メニューなどの定型行は除く
・アクセスは1秒に1回
"""
import argparse
import datetime
import os
import re
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refetch_thin import boilerplate, fetch, read_page  # noqa: E402

DEFAULT = [
    "https://www.kanazawa-it.ac.jp/about_kit/data.html",     # 数字で見るKIT
    "https://www.kanazawa-it.ac.jp/shushoku/jisseki.html",   # 学部卒業生 就職実績
    "https://www.kanazawa-it.ac.jp/career_navi/",            # 就職力ナビ
]


def norm(u):
    u = u.split("#")[0].split("?")[0]
    u = re.sub(r"(index\.html?)$", "", u)
    return u.replace("http://", "https://").rstrip("/")


def safe_name(url):
    base = re.sub(r"^https?://", "", url)
    return re.sub(r"[^0-9A-Za-z一-鿿぀-ヿ._-]+", "_", base).strip("_")[:80] + ".md"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knowledge")
    ap.add_argument("--urls", nargs="*", default=DEFAULT)
    ap.add_argument("--check", action="store_true", help="資料にあるかと文字数を見るだけ（書き換えない）")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()

    files = [f for f in os.listdir(a.knowledge) if f.endswith(".md") and f != "README.md"]
    pages = {f: read_page(os.path.join(a.knowledge, f)) for f in files}
    by_url = {norm(m.get("出典", "").split("（")[0].strip()): f for f, (m, _) in pages.items()}
    common = boilerplate(pages)

    if not a.check:
        backup = os.path.join("outputs", "knowledge_before_add")
        if not os.path.exists(backup):
            shutil.copytree(a.knowledge, backup)
            print(f"控え: {backup}")
    for url in a.urls:
        have = by_url.get(norm(url))
        old = sum(len(l) for l in pages[have][1] if l not in common) if have else 0
        try:
            lines = [l for l in fetch(url) if l not in common]
        except Exception as e:
            print(f"  ×  {url}  {type(e).__name__}")
            time.sleep(1)
            continue
        new = sum(len(l) for l in lines)
        state = f"資料にあり（{have}・{old}字）" if have else "資料に無い"
        print(f"  {state} → 取り直し {new}字  {url}")
        if not a.check and new >= 100:
            name = have or safe_name(url)
            with open(os.path.join(a.knowledge, name), "w", encoding="utf-8") as fh:
                fh.write("\n".join([f"出典: {url}", f"確認日: {a.date}", ""] + lines) + "\n")
        time.sleep(1)
    if not a.check:
        print("\n書き換えました → 次の実行で埋め込みが作り直されます（約5分・API 約300回）")


if __name__ == "__main__":
    main()
