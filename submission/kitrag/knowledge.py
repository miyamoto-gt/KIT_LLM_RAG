"""knowledge/ の .md / .txt を読み、チャンクに分ける。

各ファイルの先頭の「出典: …」「確認日: …」行を読み取り、チャンクの source にする。
knowledge/_handmade/ のファイルは手作りデータ（1ファイル＝1チャンク、検索で優先）。
"""
import re
from dataclasses import dataclass

from . import settings as S


@dataclass
class Chunk:
    id: int
    text: str            # 埋め込み・プロンプトに使う本文（先頭にタイトル行）
    source: str          # 出典 URL など
    title: str
    type: str            # web / manual
    seq: int             # 同じページの中での順番（前後をつなぐのに使う）


_META = re.compile(r"^(出典|確認日|種類)\s*[:：]\s*(.*)$")


def parse_file(path):
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta, i = {}, 0
    while i < len(lines) and (not lines[i].strip() or _META.match(lines[i].strip())):
        m = _META.match(lines[i].strip())
        if m:
            meta[m.group(1)] = m.group(2).strip()
        i += 1
    body = "\n".join(lines[i:]).strip()
    title = ""
    if body.startswith("【"):
        title = body.split("\n", 1)[0]
        body = body.split("\n", 1)[1] if "\n" in body else ""
    source = re.sub(r"（.*?）$", "", meta.get("出典", path.stem)).strip() or path.stem
    return source, title, body


def split_text(text, size=S.CHUNK_CHARS, overlap=S.CHUNK_OVERLAP):
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    out, start = [], 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            out.append(piece)
        start += size - overlap
    return out


def load_chunks(knowledge_dir=S.KNOWLEDGE_DIR):
    chunks = []
    for path in sorted(knowledge_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or path.name == "README.md":
            continue
        source, title, body = parse_file(path)
        manual = S.HANDMADE_DIR in path.parts
        pieces = [body] if manual else split_text(body)
        for seq, piece in enumerate(pieces):
            text = f"{title}\n{piece}" if title else piece
            chunks.append(Chunk(len(chunks), text, source, title, "manual" if manual else "web", seq))
    return chunks
