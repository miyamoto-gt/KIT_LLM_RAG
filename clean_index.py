"""作成済みの索引から、文字化けしたチャンクだけを取り除く（埋め込みのやり直し不要）。

使い方:  python clean_index.py
  data/index/faiss.index と data/index/chunks.jsonl を上書きする。
  元のファイルは data/index/backup/ に残す。
"""
import os
import shutil

import faiss

from config import CFG
from preprocess import is_readable
from schema import load_chunks, save_chunks


def main() -> None:
    idx_path = os.path.join(CFG.index_dir, "faiss.index")
    chunks_path = os.path.join(CFG.index_dir, "chunks.jsonl")

    backup = os.path.join(CFG.index_dir, "backup")
    os.makedirs(backup, exist_ok=True)
    for p in (idx_path, chunks_path):
        shutil.copy2(p, backup)

    index = faiss.read_index(idx_path)
    chunks = load_chunks(chunks_path)
    assert index.ntotal == len(chunks), "索引とチャンクの数が一致しません"

    keep = [i for i, c in enumerate(chunks) if is_readable(c.text)]
    vecs = index.reconstruct_n(0, index.ntotal)[keep]

    new_index = faiss.IndexFlatIP(vecs.shape[1])
    new_index.add(vecs)
    faiss.write_index(new_index, idx_path)
    save_chunks([chunks[i] for i in keep], chunks_path)

    removed = len(chunks) - len(keep)
    kept = set(keep)
    bad_sources = sorted({chunks[i].source for i in range(len(chunks)) if i not in kept})
    print(f"{len(chunks)} → {len(keep)} チャンク（文字化け {removed} 件を除外）")
    print(f"除外したチャンクを含むファイル: {len(bad_sources)} 件")
    for s in bad_sources[:20]:
        print("  ", s)


if __name__ == "__main__":
    main()