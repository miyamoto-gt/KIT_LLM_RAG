"""索引のチャンク（chunks.jsonl）を、運営テンプレートの knowledge/*.md に書き出す。

  python build/export_knowledge.py data/index/chunks.jsonl ../rag-starter-kit/my_rag/knowledge --date 2026-10-29

・ページ（出典）ごとに1ファイル。先頭に「出典URL」と「確認日」を書く（大会の資料収集方針）
・同じページのチャンクは並び順のままつなぐ。分割のときの重なり（overlap）は取り除く
・重複チャンク・文字化けしたチャンクは書き出さない
・--skip で、質問と関係の薄い区画を外せる（例: --skip kanazawa-it.ac.jp/tokyo）
"""
import argparse
import hashlib
import json
import os
import re
from collections import OrderedDict


def readable_ratio(text):
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    ok = sum(1 for c in chars if "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "　" <= c <= "〿"
             or "＀" <= c <= "￯" or " " <= c <= "~")
    return ok / len(chars)


def body_of(text):
    """先頭の【タイトル】行を除いた本文。"""
    return text.split("\n", 1)[1] if text.startswith("【") and "\n" in text else text


def merge(prev, cur, max_overlap=200):
    """前のチャンクの末尾と次のチャンクの先頭が重なっていれば、重なりを除いてつなぐ。"""
    for k in range(min(max_overlap, len(prev), len(cur)), 20, -1):
        if prev.endswith(cur[:k]):
            return prev + cur[k:]
    return prev + "\n" + cur


def safe_name(source, used):
    base = re.sub(r"^https?://", "", source)
    base = re.sub(r"[^0-9A-Za-z一-鿿぀-ヿ._-]+", "_", base).strip("_")[:80] or "source"
    name = base
    if name in used:
        name = f"{base}_{hashlib.md5(source.encode()).hexdigest()[:6]}"
    used.add(name)
    return name + ".md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out_dir")
    ap.add_argument("--date", default="2026-10-29", help="資料を確認（巡回）した日")
    ap.add_argument("--skip", nargs="*", default=[], help="書き出さない区画（URL の一部）")
    a = ap.parse_args()

    pages = OrderedDict()
    handmade = []
    seen, n_dup, n_bad, n_skip = set(), 0, 0, 0
    with open(a.chunks, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            src, text = c.get("source", ""), c.get("text", "")
            if any(s in src for s in a.skip):
                n_skip += 1
                continue
            if text in seen:
                n_dup += 1
                continue
            seen.add(text)
            if readable_ratio(body_of(text)) < 0.85:
                n_bad += 1
                continue
            if c.get("type") in ("manual", "map"):     # 手作りデータは1件ずつ別ファイル（検索で優先するため）
                handmade.append(c)
                continue
            p = pages.setdefault(src, {"title": c.get("title", ""), "head": "", "body": ""})
            if not p["head"] and text.startswith("【"):
                p["head"] = text.split("\n", 1)[0]
            b = body_of(text).strip()
            p["body"] = merge(p["body"], b) if p["body"] else b

    os.makedirs(a.out_dir, exist_ok=True)
    used, total = set(), 0
    for src, p in pages.items():
        is_url = src.startswith("http")
        lines = [f"出典: {src if is_url else src + '（金沢工業大学の公開情報をもとに手作業で書き起こし）'}",
                 f"確認日: {a.date}", ""]
        if p["head"]:
            lines.append(p["head"])
        lines.append(p["body"])
        text = "\n".join(lines).strip() + "\n"
        with open(os.path.join(a.out_dir, safe_name(src, used)), "w", encoding="utf-8") as f:
            f.write(text)
        total += len(text.encode("utf-8"))
    hm_dir = os.path.join(a.out_dir, "_handmade")
    os.makedirs(hm_dir, exist_ok=True)
    for c in handmade:
        src = c.get("source", "")
        origin = src if src.startswith("http") else f"{src}（金沢工業大学の公開情報をもとに手作業で書き起こし）"
        text = f"出典: {origin}\n確認日: {a.date}\n種類: 手作業でまとめた要約\n\n{c['text'].strip()}\n"
        name = re.sub(r"[^0-9A-Za-z_-]+", "_", c.get("id", "")) or hashlib.md5(text.encode()).hexdigest()[:8]
        with open(os.path.join(hm_dir, name + ".md"), "w", encoding="utf-8") as f:
            f.write(text)
        total += len(text.encode("utf-8"))
    print(f"手作りデータ {len(handmade)}件を _handmade/ に書き出しました")
    print(f"{len(pages)}ページを書き出しました（{total / 1024 / 1024:.1f}MB）"
          f"　除外：重複 {n_dup} / 文字化け {n_bad} / 区画指定 {n_skip}")


if __name__ == "__main__":
    main()