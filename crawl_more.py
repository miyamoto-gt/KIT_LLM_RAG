"""追加クロール：指定した分野のページから出発し、まだ索引に無いページだけをチャンクにする。

使い方:
  python crawl_more.py                         # 下の SEEDS から出発
  python crawl_more.py URL1 URL2 ...           # 出発点を指定
  python crawl_more.py --max-pages 200 --depth 2
その後:
  python add_chunks.py data/more_chunks.jsonl --replace-sources --exclude mondai_

・索引にあるページも、リンクをたどるためだけに読みに行く（保存はしない）
・取れないことが分かっているサブドメインは最初から飛ばす
・途中で止めても、それまでの分は data/more_chunks.jsonl に保存される
"""
import argparse
import os
import time
from collections import deque
from urllib.parse import urlparse

from config import CFG
from crawl import extract_links, fetch, make_session, normalize
from preprocess import document_to_chunks
from schema import load_chunks, save_chunks

# 出題されそうな分野の入口（存在しないURLは自動で飛ばす。大学サイトのメニューを見て足してよい）
SEEDS = [
    "https://www.kanazawa-it.ac.jp/nyusi/",
    "https://www.kanazawa-it.ac.jp/campuslife/",
    "https://www.kanazawa-it.ac.jp/shushoku/",
    "https://www.kanazawa-it.ac.jp/kenkyu/",
    "https://www.kanazawa-it.ac.jp/about_kit/",
    "https://www.kanazawa-it.ac.jp/kyoiku/",
    "https://www.kanazawa-it.ac.jp/gakubu_daigakuin/",
    "https://www.kanazawa-it.ac.jp/campus_guide/2026/",
    "https://www.kanazawa-it.ac.jp/access/",
]

# 前回のクロールで、証明書エラー・タイムアウト・学内専用だったサブドメイン
SKIP_HOSTS = ("europa.", "www2.", "mercury.", "navi.mars.")
SKIP_WORDS = ("mondai_", "/kifu/", "/en/", "english")   # 入試問題・寄付フォーム・英語版


def skip(url: str) -> bool:
    host = urlparse(url).netloc
    return host.startswith(SKIP_HOSTS) or any(w in url for w in SKIP_WORDS)


def main(seeds: list, max_pages: int, depth: int) -> None:
    known = {c.source for c in load_chunks(os.path.join(CFG.index_dir, "chunks.jsonl"))}
    print(f"索引にあるページ: {len(known)} 件")

    out_path = "data/more_chunks.jsonl"
    new_chunks, saved, visited = [], 0, set()
    queue = deque((normalize(u), 0) for u in seeds)
    session = make_session()
    try:
        while queue and saved < max_pages:
            url, d = queue.popleft()
            if url in visited or skip(url):
                continue
            visited.add(url)
            status = "既知" if url in known else "新規"
            print(f"[訪問 {len(visited):4d} / 新規 {saved:3d} / 待ち {len(queue):4d}] {status} {url}", flush=True)
            got = fetch(session, url)
            time.sleep(CFG.crawl_delay)
            if not got:
                print("    → 取得できず（404・対象外の形式など）", flush=True)
                continue
            kind, body = got
            if url not in known:                      # 新しいページだけ保存
                chunks = document_to_chunks(url, kind, body)
                if chunks:
                    new_chunks.extend(chunks)
                    saved += 1
                    if saved % 20 == 0:
                        save_chunks(new_chunks, out_path)
                        print(f"  新規 {saved} ページ / 訪問 {len(visited)} / 待ち {len(queue)}")
            if kind == "web" and d < depth:
                for link in extract_links(body, url):
                    if link not in visited and not skip(link):
                        queue.append((link, d + 1))
    except KeyboardInterrupt:
        print("\n中断しました。ここまでの分を保存します。")
    save_chunks(new_chunks, out_path)
    print(f"\n新規ページ {saved} 件 → {out_path}（{len(new_chunks)} チャンク）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="*")
    ap.add_argument("--max-pages", type=int, default=300, help="新しく保存するページ数の上限")
    ap.add_argument("--depth", type=int, default=2, help="出発点から何回リンクをたどるか")
    a = ap.parse_args()
    main(a.seeds or SEEDS, a.max_pages, a.depth)