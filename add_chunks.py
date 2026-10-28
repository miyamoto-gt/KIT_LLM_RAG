"""手作りのチャンク（JSONL）だけを埋め込み、既存の索引に追加する（全体の作り直し不要）。

使い方:  python add_chunks.py data/kit_facts.jsonl [ほかのファイル ...]
  同じ id のチャンクがすでにあれば、古いものを除いて入れ替える（内容を直して再実行してよい）。
  元のファイルは data/index/backup/ に残す。
"""
import os
import shutil
import sys

import faiss

from config import CFG
from index import Embedder
from schema import load_chunks, save_chunks


def main(paths: list) -> None:
    idx_path = os.path.join(CFG.index_dir, "faiss.index")
    chunks_path = os.path.join(CFG.index_dir, "chunks.jsonl")
    backup = os.path.join(CFG.index_dir, "backup")
    os.makedirs(backup, exist_ok=True)
    for p in (idx_path, chunks_path):
        shutil.copy2(p, backup)

    new = []
    for p in paths:
        new.extend(load_chunks(p))
    for c in new:
        if c.type == "web":
            c.type = "manual"
    new_ids = {c.id for c in new}

    index = faiss.read_index(idx_path)
    chunks = load_chunks(chunks_path)
    keep = [i for i, c in enumerate(chunks) if c.id not in new_ids]   # 同じidは入れ替え
    vecs = index.reconstruct_n(0, index.ntotal)[keep]

    new_vecs = Embedder().docs([c.text for c in new])
    out = faiss.IndexFlatIP(vecs.shape[1])
    out.add(vecs)
    out.add(new_vecs)
    faiss.write_index(out, idx_path)
    save_chunks([chunks[i] for i in keep] + new, chunks_path)
    print(f"{len(chunks)} → {len(keep) + len(new)} チャンク（入れ替え {len(chunks) - len(keep)} 件、追加 {len(new)} 件）")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("使い方: python add_chunks.py data/kit_facts.jsonl")
    main(sys.argv[1:])