"""資料（knowledge/）が、サイトのどの区画にどれだけあるかを調べる。足りない区画を見つけるため。

  python build/data_coverage.py submission/knowledge                 # 区画ごとのページ数・文字数・チャンク数（API もネットも使わない）
  python build/data_coverage.py submission/knowledge --sitemap       # 大学のサイトマップと比べて、取れていないページを区画ごとに出す（ネットを使う）

・区画は URL の最初の1〜2階層（例: kanazawa-it.ac.jp/nyusi、kanazawa-it.ac.jp/campus_guide/2026）
・「話題」は区画を大きくまとめたもの（大学案内・学部大学院・入試・学生生活・就職・国際・研究・ニュース など）
・中身が薄い（定型行を除いて300字未満）ページの数も出す（取り直しの候補）
・結果は outputs/data_coverage.txt にも書く
"""
import argparse
import collections
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse

TOPICS = [   # (話題, URL に含まれる文字列)。上から順に当てはめる
    ("古い年度の便覧", ["/campus_guide/2021", "/campus_guide/2022", "/campus_guide/2023", "/campus_guide/2024",
                      "/campus_guide/2025"]),
    ("教員名簿", ["/kyouinroku"]),
    ("国際", ["/international", "/ekit"]),
    ("大学案内", ["/about_kit"]),
    ("学部・大学院", ["/gakubu_daigakuin", "/campus_guide/2026/chapter_6", "/kisoku"]),
    ("入試", ["/nyusi", "keinet"]),
    ("学生生活・施設", ["/campuslife", "/gstation", "/shisetsu", "/kitlc", "/hogosha", "/safety", "/campus_guide/2026/chapter_2",
                       "/campus_guide/2026/chapter_5", "/campus_guide/2026/campus_map", "キャンパスマップ"]),
    ("学習・教育", ["/kyoiku", "/campus_guide/2026/chapter_1", "/campus_guide/2026/chapter_3", "/campus_guide/2026/chapter_4",
                   "/kyoshoku", "w3e.kanazawa-it", "/cdio"]),
    ("就職・キャリア", ["/shushoku", "/career", "/intern/", "/coop"]),
    ("研究・産学連携", ["/kenkyu", "/research", "/sangaku", "/jigyo", "/ls-pbl", "/sdgs"]),
    ("課外活動・夢考房", ["/yumekobo", "/project", "/gakuyukai", "/syouyukai"]),
    ("ニュース", ["/kitnews", "/news"]),
    ("学費・奨学金・寄付", ["/kifu", "shogakukin", "/houjin"]),
    ("その他キャンパス", ["/tokyo", "/hakusanroku", "/yatsukaho"]),
    ("CAMPUS GUIDE 2026（その他の章・PDF）", ["/campus_guide/2026"]),
]


def topic_of(src):
    for name, keys in TOPICS:
        if any(k in src for k in keys):
            return name
    return "その他"


def section_of(src, depth=2, hand=False):
    if hand:
        return "(手作りデータ)"
    if not src.startswith("http"):
        return f"(手作り) {src[:20]}"
    u = urlparse(src)
    parts = [p for p in u.path.split("/") if p and "." not in p][:depth]
    return u.netloc.replace("www.", "") + ("/" + "/".join(parts) if parts else "")


def read(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"^出典\s*[:：]\s*(\S+)", t, re.M)
    src = m.group(1) if m else ""
    body = "\n".join(l for l in t.split("\n") if not re.match(r"^(出典|確認日|種類)\s*[:：]", l))
    return src, body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knowledge")
    ap.add_argument("--sitemap", action="store_true", help="サイトマップと比べる（ネットを使う）")
    ap.add_argument("--depth", type=int, default=2, help="区画の階層の深さ")
    a = ap.parse_args()

    rows = []   # (src, topic, section, chars, handmade)
    for root, _, files in os.walk(a.knowledge):
        for f in files:
            if not f.endswith((".md", ".txt")) or f == "README.md":
                continue
            src, body = read(os.path.join(root, f))
            hand = "_handmade" in root
            rows.append((src, "手作りデータ" if hand else topic_of(src), section_of(src, a.depth, hand), len(body), hand))
    # 定型行（多くのページに出る短い行）を除いた本文の長さ → 薄いページ
    cnt = collections.Counter()
    bodies = {}
    for root, _, files in os.walk(a.knowledge):
        for f in files:
            if f.endswith(".md") and f != "README.md" and "_handmade" not in root:
                src, body = read(os.path.join(root, f))
                lines = {l.strip() for l in body.split("\n") if l.strip()}
                bodies[src] = lines
                cnt.update(lines)
    common = {l for l, c in cnt.items() if c >= 50 and len(l) <= 40}
    thin = {s for s, ls in bodies.items() if sum(len(l) for l in ls if l not in common) < 300}

    out = []
    p = out.append
    total_chars = sum(r[3] for r in rows)
    p(f"資料 {len(rows)} ファイル・{total_chars / 10000:.0f}万字・チャンクの目安 {total_chars // 400}（500字・重なり100字）")
    p("\n【話題ごと】ページ数・文字数・割合・中身が薄いページ")
    by_topic = collections.defaultdict(lambda: [0, 0, 0])
    for src, tp, sec, ch, hand in rows:
        by_topic[tp][0] += 1
        by_topic[tp][1] += ch
        by_topic[tp][2] += src in thin
    for tp, (n, ch, th) in sorted(by_topic.items(), key=lambda x: -x[1][1]):
        bar = "■" * max(1, round(40 * ch / total_chars))
        p(f"  {tp:14s} {n:5d}ページ {ch / 10000:7.1f}万字 {100 * ch / total_chars:5.1f}%  薄い{th:3d}  {bar}")
    p(f"\n【区画ごと】（上位40）")
    by_sec = collections.defaultdict(lambda: [0, 0, ""])
    for src, tp, sec, ch, hand in rows:
        by_sec[sec][0] += 1
        by_sec[sec][1] += ch
        by_sec[sec][2] = tp
    for sec, (n, ch, tp) in sorted(by_sec.items(), key=lambda x: -x[1][1])[:40]:
        p(f"  {n:5d}ページ {ch / 10000:6.1f}万字  [{tp}] {sec}")

    if a.sitemap:
        p("\n【サイトマップとの比較】")
        try:
            import requests
            from bs4 import BeautifulSoup
            have = {re.sub(r"(index\.html?)?$", "", r[0].split("#")[0]).rstrip("/") for r in rows}
            base = "https://www.kanazawa-it.ac.jp/sitemap/"
            html = requests.get(base, timeout=20, headers={"User-Agent": "KIT-RAG-study (student competition)"}).text
            links = set()
            for a_ in BeautifulSoup(html, "lxml").find_all("a", href=True):
                u = urljoin(base, a_["href"]).split("#")[0]
                if "kanazawa-it.ac.jp" in u and not u.lower().endswith((".jpg", ".png", ".zip")):
                    links.add(re.sub(r"(index\.html?)?$", "", u).rstrip("/"))
            missing = sorted(l for l in links if l not in have and l.replace("https://", "http://") not in have)
            p(f"  サイトマップのリンク {len(links)} 件・資料に無いもの {len(missing)} 件")
            by = collections.defaultdict(list)
            for l in missing:
                by[topic_of(l)].append(l)
            for tp, ls in sorted(by.items(), key=lambda x: -len(x[1])):
                p(f"  [{tp}] {len(ls)} 件")
                for l in ls[:15]:
                    p(f"      {l}")
                if len(ls) > 15:
                    p(f"      … ほか {len(ls) - 15} 件")
        except Exception as e:
            p(f"  サイトマップを取れませんでした: {type(e).__name__}: {e}")

    text = "\n".join(out)
    print(text)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/data_coverage.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n→ outputs/data_coverage.txt")


if __name__ == "__main__":
    main()