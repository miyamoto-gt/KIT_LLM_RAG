"""中身が薄いページ（本文の取り出しに失敗したページ）を見つけて、取り直す。

巡回のときの本文抽出で、表・<dl>・数字の並びなどが落ちて、メニューだけが残ったページがある
（例：about_kit/gakuseisuu.html は「学部学生数 5,780名・大学院生 551名」が抜けていた）。
このスクリプトは knowledge/ の各ページから「多くのページに出てくる定型行（メニュー）」を除いた本文の長さを測り、
短いページの URL をもう一度取りにいって、表を「見出し | 値」の行にした本文で書き直す。

使い方（KIT_LLM_RAG の直下から。ネットワークが必要なので手元の Mac で実行する）:
  python build/refetch_thin.py submission/knowledge --list                 # 薄いページの一覧だけ（取りにいかない）
  python build/refetch_thin.py submission/knowledge --dry-run              # 取り直して、増える文字数だけ表示（書き換えない）
  python build/refetch_thin.py submission/knowledge                        # 取り直して、長くなったページだけ書き換える
  python build/refetch_thin.py submission/knowledge --urls https://www.kanazawa-it.ac.jp/about_kit/gakuseisuu.html

・書き換える前の knowledge は outputs/knowledge_before_refetch/ に控える
・書き換えたファイルの「確認日」は今日の日付になる（大会の資料収集方針：出典URLと確認日）
・画像だけの PDF（スキャン）は文字が取れないので、一覧に「画像のみ」と出す（手作りデータで補う候補）
・アクセスは1秒に1回まで（相手のサイトに負担をかけない）
"""
import argparse
import collections
import datetime
import io
import os
import re
import shutil
import sys
import time

HEADER = re.compile(r"^(出典|確認日|種類)\s*[:：]")
NAV_HINT = re.compile(r"(^|[\s_-])(nav|gnav|lnav|snav|menu|breadcrumb|footer|header|sidebar|pagetop|sitemap|search)([\s_-]|$)", re.I)


def read_page(path):
    lines = open(path, encoding="utf-8").read().split("\n")
    meta = {}
    body = []
    for l in lines:
        m = re.match(r"^(出典|確認日|種類)\s*[:：]\s*(.*)$", l)
        if m and not body:
            meta[m.group(1)] = m.group(2).strip()
        else:
            body.append(l)
    return meta, [l.strip() for l in body if l.strip()]


def boilerplate(pages, min_pages=50, max_len=40):
    """多くのページに出てくる短い行（メニュー・フッター）。"""
    cnt = collections.Counter()
    for _, lines in pages.values():
        cnt.update(set(lines))
    return {l for l, c in cnt.items() if c >= min_pages and len(l) <= max_len}


def content_len(lines, common):
    return sum(len(l) for l in lines if l not in common and not l.startswith("【"))


# ---------- 取り直し ----------
def html_to_text(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "iframe", "svg", "form", "nav", "header", "footer", "aside"]):
        t.decompose()
    for t in soup.find_all(True):
        if t.attrs is None:
            continue
        key = " ".join([t.get("id") or ""] + list(t.get("class") or []))
        if key and NAV_HINT.search(key):
            t.decompose()
    # 表は1行ずつ「セル | セル」にする（見出し行も残す）
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [" ".join(c.get_text(" ", strip=True).split()) for c in tr.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        table.replace_with("\n" + "\n".join(rows) + "\n")
    # <dl> は「見出し：中身」
    for dl in soup.find_all("dl"):
        items = []
        for dt in dl.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            a = " ".join(dt.get_text(" ", strip=True).split())
            b = " ".join(dd.get_text(" ", strip=True).split()) if dd else ""
            items.append(f"{a}：{b}" if b else a)
        dl.replace_with("\n" + "\n".join(items) + "\n")
    main = soup.find("main") or soup.find(id=re.compile("main|content|contents", re.I)) or soup.body or soup
    text = main.get_text("\n")
    lines = [" ".join(l.split()) for l in text.split("\n")]
    return [l for l in lines if l]


def pdf_to_text(data):
    from pypdf import PdfReader
    r = PdfReader(io.BytesIO(data))
    lines = []
    for p in r.pages:
        t = p.extract_text() or ""
        lines += [" ".join(l.split()) for l in t.split("\n") if l.strip()]
    return lines


def fetch(url, timeout=20):
    import requests
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "KIT-RAG-study (student competition)"})
    r.raise_for_status()
    if url.lower().endswith(".pdf") or "pdf" in r.headers.get("Content-Type", ""):
        return pdf_to_text(r.content)
    r.encoding = r.apparent_encoding or r.encoding
    return html_to_text(r.text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("knowledge")
    ap.add_argument("--min-chars", type=int, default=300, help="定型行を除いた本文がこれ未満なら「薄い」")
    ap.add_argument("--list", action="store_true", help="一覧だけ出して終わる")
    ap.add_argument("--dry-run", action="store_true", help="取り直すが書き換えない")
    ap.add_argument("--urls", nargs="*", help="このURLのページだけ取り直す（薄くなくても）")
    ap.add_argument("--skip", nargs="*", default=["kyouinroku", "kitcr", "sitemap", "yumekobo/news/data"],
                    help="取り直さない URL の一部（教員名簿・スキャンの広報誌など）")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    a = ap.parse_args()

    files = [f for f in sorted(os.listdir(a.knowledge)) if f.endswith(".md") and f != "README.md"]
    pages = {f: read_page(os.path.join(a.knowledge, f)) for f in files}
    common = boilerplate(pages)
    targets = []
    for f, (meta, lines) in pages.items():
        url = meta.get("出典", "").split("（")[0].strip()
        if not url.startswith("http"):
            continue
        n = content_len(lines, common)
        if a.urls is not None:
            if url in a.urls:
                targets.append((f, url, n))
        elif n < a.min_chars and not any(s in url for s in a.skip):
            targets.append((f, url, n))
    targets.sort(key=lambda t: t[2])
    print(f"ページ {len(files)} 件・定型行 {len(common)} 種・取り直す候補 {len(targets)} 件")
    if a.list:
        for f, url, n in targets:
            print(f"  {n:5d}字  {url}")
        return

    if not a.dry_run:
        backup = os.path.join("outputs", "knowledge_before_refetch")
        if not os.path.exists(backup):
            shutil.copytree(a.knowledge, backup)
            print(f"控え: {backup}")
    improved = failed = same = 0
    for f, url, n in targets:
        try:
            new = fetch(url)
        except Exception as e:
            print(f"  ×  {url}  {type(e).__name__}: {str(e)[:60]}")
            failed += 1
            time.sleep(1)
            continue
        new = [l for l in new if l not in common]
        m = sum(len(l) for l in new)
        if m <= n + 50:
            print(f"  ・ {n:5d}字 → {m:5d}字  {url}" + ("  （画像のみ？）" if url.lower().endswith(".pdf") and m < 50 else ""))
            same += 1
        else:
            print(f"  ○ {n:5d}字 → {m:5d}字  {url}")
            improved += 1
            if not a.dry_run:
                meta, old = pages[f]
                title = next((l for l in old if l.startswith("【")), "")
                text = "\n".join([f"出典: {url}", f"確認日: {a.date}", ""] + ([title] if title else []) + new) + "\n"
                with open(os.path.join(a.knowledge, f), "w", encoding="utf-8") as fh:
                    fh.write(text)
        time.sleep(1)
    print(f"\n長くなった {improved} 件・変わらない {same} 件・取れなかった {failed} 件"
          + ("（--dry-run なので書き換えていません）" if a.dry_run else ""))
    if improved and not a.dry_run:
        print("→ 資料が変わったので、埋め込みは次の実行で作り直されます（手元は約4分）")


if __name__ == "__main__":
    main()