"""手作りのチャンク（JSONL）だけを埋め込み、既存の索引に追加する（全体の作り直し不要）。

使い方:
  python add_chunks.py data/kit_facts.jsonl [ほかのファイル ...]
  python add_chunks.py data/pdf_fixed.jsonl --replace-sources --exclude mondai_

  同じ id のチャンクがすでにあれば、古いものを除いて入れ替える（内容を直して再実行してよい）。
  --replace-sources : 追加するチャンクと同じURLの古いチャンクをすべて消してから入れる（読み直したPDF用）
  --exclude 文字列  : URLにその文字列を含むチャンクは追加しない（複数指定可）
  元のファイルは data/index/backup/ に残す。
"""
import argparse
import os
import shutil

from config import CFG
from index import Embedder
from schema import load_chunks, save_chunks

# faiss は埋め込みの計算が終わってから読み込む。
# Mac では faiss と torch の並列計算部品（OpenMP）がぶつかり、計算中に落ちることがあるため。


def main(paths: list, replace_sources: bool = False, exclude: list = ()) -> None:
    idx_path = os.path.join(CFG.index_dir, "faiss.index")
    chunks_path = os.path.join(CFG.index_dir, "chunks.jsonl")
    backup = os.path.join(CFG.index_dir, "backup")
    os.makedirs(backup, exist_ok=True)
    for p in (idx_path, chunks_path):
        shutil.copy2(p, backup)

    new = []
    for p in paths:
        new.extend(load_chunks(p))
    if exclude:
        before = len(new)
        new = [c for c in new if not any(x in c.source for x in exclude)]
        print(f"除外指定により {before - len(new)} チャンクを追加しません")
    for c in new:
        if c.type == "web":
            c.type = "manual"
    new_ids = {c.id for c in new}
    new_sources = {c.source for c in new} if replace_sources else set()

    new_vecs = Embedder().docs([c.text for c in new])   # ← 先に計算する

    import faiss
    index = faiss.read_index(idx_path)
    chunks = load_chunks(chunks_path)
    keep = [i for i, c in enumerate(chunks)
            if c.id not in new_ids and c.source not in new_sources]   # 同じid（と指定時は同じURL）は入れ替え
    vecs = index.reconstruct_n(0, index.ntotal)[keep]

    out = faiss.IndexFlatIP(vecs.shape[1])
    out.add(vecs)
    out.add(new_vecs)
    faiss.write_index(out, idx_path)
    save_chunks([chunks[i] for i in keep] + new, chunks_path)
    print(f"{len(chunks)} → {len(keep) + len(new)} チャンク（入れ替え {len(chunks) - len(keep)} 件、追加 {len(new)} 件）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--replace-sources", action="store_true")
    ap.add_argument("--exclude", action="append", default=[])
    a = ap.parse_args()
    main(a.paths, a.replace_sources, a.exclude)