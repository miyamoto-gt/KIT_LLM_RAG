"""②③ 前処理とチャンク化：保存したHTML/PDFから本文を抜き出し、共通形式のチャンクにする。"""
import hashlib
import io
import json
import os

from bs4 import BeautifulSoup

from config import CFG
from schema import Chunk, load_chunks, save_chunks

BREADCRUMB_SEL = ('[class*="breadcrumb"], [id*="breadcrumb"], [class*="pankuzu"], '
                  '[id*="pankuzu"], [class*="topicpath"], [id*="topicpath"]')


def html_to_text(html: str):
    """(タイトル, パンくず, 本文) を返す。本文はメニュー・フッターを除いたもの。"""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    bc = soup.select_one(BREADCRUMB_SEL)
    breadcrumb = bc.get_text(" > ", strip=True) if bc else ""

    body = None
    try:
        import trafilatura
        try:   # 表を Markdown で残せる版なら Markdown
            body = trafilatura.extract(html, include_tables=True, output_format="markdown")
        except Exception:
            body = trafilatura.extract(html, include_tables=True)
    except ImportError:
        pass
    if not body:  # 抽出失敗時の保険：main要素 or body全体
        for t in soup(["script", "style", "nav", "header", "footer", "noscript"]):
            t.decompose()
        main = soup.find("main") or soup.body or soup
        body = main.get_text("\n", strip=True)
    return title, breadcrumb, body


READABLE_EXTRA = set("・―‐…“”‘’「」『』【】〜")


def is_readable(text: str, min_ratio: float = 0.85) -> bool:
    """日本語・英数字として読める文字の割合で、文字化けしたテキストを判定する。

    フォントに文字の対応表が無いPDFは、ギリシャ文字やアラビア文字のような記号に化けるので、
    ひらがな・カタカナ・漢字・英数字・全角記号以外が多いものを除外する。
    """
    body = text.split("\n", 1)[-1] if text.startswith("【") else text
    chars = [c for c in body if not c.isspace()]
    if len(chars) < 20:
        return len(chars) > 0
    ok = sum(1 for c in chars if
             "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "　" <= c <= "〿"
             or "＀" <= c <= "￯" or " " <= c <= "~" or c in READABLE_EXTRA)
    return ok / len(chars) >= min_ratio


def pdf_to_text(data: bytes) -> str:
    """PyMuPDF があれば優先（日本語PDFに強い）。無ければ pypdf。"""
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf   # 古い版の名前
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        if is_readable(text):
            return text
    except Exception:
        pass
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def split_text(text: str, size: int, overlap: int) -> list:
    """段落単位で size 字前後にまとめる。長すぎる段落は強制分割。"""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf = buf[-overlap:]
        buf = f"{buf}\n{p}" if buf else p
        while len(buf) > size * 1.5:
            chunks.append(buf[:size])
            buf = buf[size - overlap:]
    if buf:
        chunks.append(buf)
    return chunks


def make_chunks(url: str, kind: str, title: str, breadcrumb: str, body: str) -> list:
    header = f"【{title}】" + (f"（{breadcrumb}）" if breadcrumb else "")
    out = []
    for i, piece in enumerate(split_text(body, CFG.chunk_size, CFG.chunk_overlap)):
        if not is_readable(piece):        # 文字化けしたチャンクは索引に入れない
            continue
        cid = hashlib.md5(f"{url}#{i}".encode()).hexdigest()[:12]
        out.append(Chunk(id=cid, text=f"{header}\n{piece}", source=url,
                         title=title, breadcrumb=breadcrumb, type=kind))
    return out


def document_to_chunks(url: str, kind: str, body) -> list:
    """HTML文字列 or PDFバイト列 → チャンク一覧。Web補完からも使う。"""
    if kind == "pdf":
        text = pdf_to_text(body)
        title, breadcrumb = os.path.basename(url), ""
    else:
        title, breadcrumb, text = html_to_text(body)
    return make_chunks(url, kind, title, breadcrumb, text) if text.strip() else []


def build() -> None:
    chunks, seen_text = [], set()
    with open(os.path.join(CFG.raw_dir, "manifest.jsonl"), encoding="utf-8") as mf:
        for line in mf:
            m = json.loads(line)
            mode = "rb" if m["type"] == "pdf" else "r"
            with open(m["path"], mode, **({} if mode == "rb" else {"encoding": "utf-8"})) as f:
                body = f.read()
            try:
                for c in document_to_chunks(m["url"], m["type"], body):
                    key = c.text.split("\n", 1)[-1]   # 共通ヘッダを除いた本文で重複判定
                    if key not in seen_text:
                        seen_text.add(key)
                        chunks.append(c)
            except Exception as e:
                print("skip", m["url"], e)

    for path in CFG.extra_chunk_files:            # キャンパスマップ等の手作りデータ
        if os.path.exists(path):
            extra = load_chunks(path)
            for c in extra:
                c.type = c.type if c.type != "web" else "map"
            chunks.extend(extra)
            print(f"+ {len(extra)} chunks from {path}")

    save_chunks(chunks, CFG.chunks_path)
    print(f"done: {len(chunks)} chunks -> {CFG.chunks_path}")


if __name__ == "__main__":
    build()