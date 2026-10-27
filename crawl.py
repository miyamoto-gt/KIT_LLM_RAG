"""① 収集：トップページからリンクを幅優先でたどり、HTML/PDFを保存する。

木構造を後で使えるよう、各ページの「親URL」と「深さ」を manifest.jsonl に残す。
"""
import hashlib
import json
import os
import time
from collections import deque
from urllib import robotparser
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from KIT_LLM_RAG.config import CFG

SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip", ".mp4", ".mp3",
            ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx", ".css", ".js", ".ics")


def normalize(url: str) -> str:
    url, _ = urldefrag(url)                      # #以降を除去
    if url.endswith("/index.html"):
        url = url[: -len("index.html")]
    return url


def allowed(url: str) -> bool:
    p = urlparse(url)
    return (p.scheme in ("http", "https")
            and p.netloc.endswith(CFG.allowed_domain)
            and not p.path.lower().endswith(SKIP_EXT))


def extract_links(html: str, base_url: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        u = normalize(urljoin(base_url, a["href"]))
        if allowed(u):
            links.append(u)
    return links


def fetch(session: requests.Session, url: str):
    """1ページ取得して (種類, 本文) を返す。対象外なら None。Web補完からも使う。"""
    try:
        r = session.get(url, timeout=15)
    except requests.RequestException as e:
        print("NG", url, e)
        return None
    if r.status_code != 200:
        return None
    ctype = r.headers.get("Content-Type", "")
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return "pdf", r.content
    if "html" not in ctype:
        return None
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding
    return "web", r.text


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "KIT-RAG-student-crawler (educational use)"
    return s


def crawl() -> None:
    os.makedirs(CFG.raw_dir, exist_ok=True)
    rp = robotparser.RobotFileParser(urljoin(CFG.start_urls[0], "/robots.txt"))
    try:
        rp.read()
    except Exception:
        rp = None

    queue = deque((normalize(u), None, 0) for u in CFG.start_urls)
    seen = {u for u, _, _ in queue}
    session = make_session()
    n = 0
    with open(os.path.join(CFG.raw_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
        while queue and n < CFG.max_pages:
            url, parent, depth = queue.popleft()
            if rp and not rp.can_fetch("*", url):
                continue
            got = fetch(session, url)
            time.sleep(CFG.crawl_delay)
            if got is None:
                continue
            kind, body = got
            path = os.path.join(CFG.raw_dir, hashlib.md5(url.encode()).hexdigest()
                                + (".pdf" if kind == "pdf" else ".html"))
            if kind == "pdf":
                open(path, "wb").write(body)
            else:
                open(path, "w", encoding="utf-8").write(body)
            mf.write(json.dumps({"url": url, "path": path, "parent": parent,
                                 "depth": depth, "type": kind}, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"{n} pages / queue {len(queue)}")
            if kind == "web" and depth < CFG.max_depth:
                for u in extract_links(body, url):
                    if u not in seen:
                        seen.add(u)
                        queue.append((u, url, depth + 1))
    print(f"done: {n} pages saved to {CFG.raw_dir}")


if __name__ == "__main__":
    crawl()
