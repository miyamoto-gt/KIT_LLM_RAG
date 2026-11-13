"""faiss 索引のベクトルを float32 → float16 にして、ファイルサイズを約半分にする。

使い方（KIT_LLM_RAG の直下で）:
    python -m build.shrink_index

  - 元の索引は data/index/faiss_fp32.index として残す（戻すときはこれを faiss.index に戻す）
  - 変換後、ランダムな100件のベクトルで検索して、上位5件が元とどれだけ一致するかを表示する
  - faiss.read_index / index.search の使い方は変わらないので、index.py の修正は不要
"""
import os
import shutil

import faiss
import numpy as np

SRC = "data/index/faiss.index"
KEEP = "data/index/faiss_fp32.index"

if not os.path.exists(KEEP):
    shutil.copy2(SRC, KEEP)

orig = faiss.read_index(KEEP)
n, d = orig.ntotal, orig.d
print(f"元の索引: {type(orig).__name__}  件数={n}  次元={d}  metric={orig.metric_type}")

vecs = orig.reconstruct_n(0, n).astype(np.float32)

small = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_fp16, orig.metric_type)
small.train(vecs)
small.add(vecs)
faiss.write_index(small, SRC)

# 検索結果がどれだけ変わらないかを確認
rng = np.random.default_rng(0)
q = vecs[rng.choice(n, size=min(100, n), replace=False)]
_, a = orig.search(q, 5)
_, b = small.search(q, 5)
overlap = np.mean([len(set(x) & set(y)) / 5 for x, y in zip(a, b)])

mb = lambda p: os.path.getsize(p) / 1024 / 1024
print(f"サイズ: {mb(KEEP):.1f}MB → {mb(SRC):.1f}MB")
print(f"上位5件の一致率: {overlap * 100:.1f}%（100%に近ければ検索結果はほぼ変わらない）")
