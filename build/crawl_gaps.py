"""資料が薄い区画だけをたどり、まだ knowledge/ に無いページを取ってきて .md で足す。

サイトマップ（上の階層）はほぼ取れていたので、抜けているのは「区画の中の下の階層」のページ。
区画の入口ページから、同じ区画の中のリンクを2階層までたどり、資料に無いものを一覧・取得する。

  python build/crawl_gaps.py submission/knowledge --list        # 取っていないページの一覧だけ（入口ページは取りにいく）
  python build/crawl_gaps.py submission/knowledge               # 取ってきて knowledge/ に .md で足す
  python build/crawl_gaps.py submission/knowledge --sections /campuslife/ /shushoku/   # 区画を指定

・取る前の knowledge は outputs/knowledge_before_gaps/ に控える
・本文が200字未満のページ（一覧・リンクだけ）は足さない
・アクセスは1秒に1回まで。上限は --max（既定 200 ページ）
・本文の取り出し方は refetch_thin.py と同じ（表は「| 区切り」、<dl> は「見出し：中身」）
"""
import argparse
import datetime
import os
import re
import shutil
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refetch_thin import boilerplate, fetch, read_page  # noqa: E402

BASE = "https://www.kanazawa-it.ac.jp"
SECTIONS = [   # 資料が薄かった区画（11/21 の data_coverage.py の結果から）
    "/campuslife/", "/gstation/", "/shisetsu/", "/shushoku/", "/career/", "/kenkyu/",
    "/international/", "/kifu/", "/hogosha/", "/kitlc/", "/about_kit/",
]
# 足さないページ（イベントのチラシ・データベースの使い方の PDF・古い年度・決算・音声など）。11/21 の --list の結果から
EXCLUDE = (r"/career/data/|/kitlc/page/download/|/kitlc/page/local/(movie|pdf)/|/kitlc/info/news/20(1\d|2[0-5])/"
           r"|ir_syugakushien-r[1-6]|kessan_|/event/pdf/|/internship/pdf/|sitemap|news_archive|kifumoushikomi"
           r"|europa\.|kitlc-form|\.mp3$")
SKIP = re.compile(r"\.(jpg|jpeg|png|gif|zip|mp4|xlsx?|docx?|pptx?)$|/kyouinroku|/en/|/english|mailto:|javascript:", re.I)


def norm(u):
    u = u.split("#")[0].split("?")[0]
    u = re.sub(r"(index\.html?)$", "", u)
    return u.replace("http://", "https://").rstrip("/")


def links_of(url):
    import requests
    from bs4 import BeautifulSoup
    r = requests.get(url, timeout=20, headers={"User-Agent": "KIT-RAG-study (student competition)"})
    r.raise_for_status()
    if "html" not in r.headers.get("Content-Type", "html"):
        return []
    soup = BeautifulSoup(r.text, "lxml")
    return [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]


def safe_name(url):
    base = re.sub(r"^https?://", "", url)
    return re.sub(r"[^0-9A-Za-z一-鿿぀-ヿ._-]+", "_", base).strip("_")[:80] + ".md"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knowledge")
    ap.add_argument("--sections", nargs="*", default=SECTIONS)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--max", type=int, default=200, help="取りにいくページ数の上限")
    ap.add_argument("--list", action="store_true", help="一覧だけ（足さない）")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--exclude", default=EXCLUDE, help="足さないページの URL（正規表現）。空にすると全部")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()

    files = [f for f in os.listdir(a.knowledge) if f.endswith(".md") and f != "README.md"]
    pages = {f: read_page(os.path.join(a.knowledge, f)) for f in files}
    have = {norm(m.get("出典", "").split("（")[0].strip()) for m, _ in pages.values()}
    common = boilerplate(pages)

    # 区画の入口からたどって、資料に無いページを集める
    queue = deque((BASE + s, 0) for s in a.sections)
    seen = {norm(u) for u, _ in queue}
    missing = []
    visits = 0
    while queue and visits < a.max:
        url, d = queue.popleft()
        if norm(url) not in have and norm(url) not in [norm(m) for m in missing]:
            missing.append(url)
        if d >= a.depth or url.lower().endswith(".pdf"):
            continue
        try:
            ls = links_of(url)
        except Exception as e:
            print(f"  ×  {url}  {type(e).__name__}")
            continue
        finally:
            visits += 1
            time.sleep(1)
        for u in ls:
            nu = norm(u)
            if nu in seen or SKIP.search(u) or "kanazawa-it.ac.jp" not in urlparse(u).netloc:
                continue
            if not any(s.rstrip("/") in urlparse(u).path for s in a.sections):
                continue
            seen.add(nu)
            queue.append((u, d + 1))
    ex = re.compile(a.exclude) if a.exclude else None
    skipped = [u for u in missing if ex and ex.search(u)]
    missing = [u for u in missing if not (ex and ex.search(u))]
    print(f"区画 {len(a.sections)}・たどったページ {visits}・資料に無いページ {len(missing) + len(skipped)}"
          f"（うち足さない {len(skipped)}・足す候補 {len(missing)}）")
    for u in missing:
        print(f"  {u}")
    if a.list or not missing:
        return

    backup = os.path.join("outputs", "knowledge_before_gaps")
    if not os.path.exists(backup):
        shutil.copytree(a.knowledge, backup)
        print(f"\n控え: {backup}")
    added = 0
    for url in missing[:a.max]:
        try:
            lines = [l for l in fetch(url) if l not in common]
        except Exception as e:
            print(f"  ×  {url}  {type(e).__name__}")
            time.sleep(1)
            continue
        n = sum(len(l) for l in lines)
        if n < a.min_chars:
            print(f"  ・ {n:5d}字（短いので足さない） {url}")
        else:
            name = safe_name(url)
            with open(os.path.join(a.knowledge, name), "w", encoding="utf-8") as fh:
                fh.write("\n".join([f"出典: {url}", f"確認日: {a.date}", ""] + lines) + "\n")
            print(f"  ○ {n:5d}字  {url}")
            added += 1
        time.sleep(1)
    print(f"\n足したページ {added} 件 → 次の実行で埋め込みが作り直されます")


if __name__ == "__main__":
    main()