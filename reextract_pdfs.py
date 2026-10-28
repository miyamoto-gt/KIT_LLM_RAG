"""文字化けで除外したPDFを取り直し、PyMuPDF で読めたものだけをチャンクにする。

使い方:
  1) Googleドライブの kit_rag_data/chunks.jsonl（文字化けを除く前の元データ）を
     data/chunks_original.jsonl としてダウンロードしておく
  2) python reextract_pdfs.py data/chunks_original.jsonl
  3) python add_chunks.py data/pdf_fixed.jsonl      ← 読めた分を索引に追加

結果:
  data/pdf_fixed.jsonl   … 読み直せたPDFのチャンク
  data/pdf_need_ocr.txt  … PyMuPDFでも化けたPDFのURL（OCRが必要）
"""
import sys
import time

from crawl import fetch, make_session
from config import CFG
from preprocess import is_readable, make_chunks
from schema import load_chunks, save_chunks


def find_bad_pdfs(path: str) -> list:
    """文字化けしたチャンクを1つでも含むPDFのURLを集める。"""
    bad = []
    for c in load_chunks(path):
        if c.type == "pdf" and not is_readable(c.text) and c.source not in bad:
            bad.append(c.source)
    return bad


def extract_with_pymupdf(data: bytes) -> str:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf   # 古い版の名前
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:      # 壊れたPDFなど
        print("  PDFを開けません:", type(e).__name__)
        return ""


def main(original_path: str) -> None:
    urls = find_bad_pdfs(original_path)
    print(f"文字化けしていたPDF: {len(urls)} 件")
    session = make_session()
    fixed, need_ocr = [], []
    for url in urls:
        got = fetch(session, url)
        time.sleep(CFG.crawl_delay)
        if not got or got[0] != "pdf":
            print("  取得失敗:", url)
            need_ocr.append(url)
            continue
        text = extract_with_pymupdf(got[1])
        chunks = make_chunks(url, "pdf", url.rsplit("/", 1)[-1], "", text)   # 化けたチャンクはここで落ちる
        total = len([p for p in text.split("\n") if p.strip()])
        if chunks:
            fixed.extend(chunks)
            print(f"  読めた: {len(chunks):3d} チャンク  {url}")
        else:
            need_ocr.append(url)
            print(f"  まだ化ける（OCRが必要）: {url}  （抽出行数 {total}）")

    save_chunks(fixed, "data/pdf_fixed.jsonl")
    with open("data/pdf_need_ocr.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(need_ocr) + ("\n" if need_ocr else ""))
    print(f"\n読み直せたPDF: {len(urls) - len(need_ocr)} 件 → data/pdf_fixed.jsonl（{len(fixed)} チャンク）")
    print(f"OCRが必要なPDF: {len(need_ocr)} 件 → data/pdf_need_ocr.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("使い方: python reextract_pdfs.py data/chunks_original.jsonl")
    main(sys.argv[1])