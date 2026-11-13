#!/bin/bash
# 提出用 zip（outputs/code.zip、上限30MB）を作る。KIT_LLM_RAG の直下で実行する:  bash make_zip.sh
# 実行時に必要なのは「直下のコード」と「data/index/（chunks.jsonl + faiss.index）」と小さい補助データだけ。
# build/ の .py は web_fallback.py が import するので入れる（main.py では Web補完は無効）。
set -e
mkdir -p outputs
rm -f outputs/code.zip
zip -r -9 outputs/code.zip . \
  -x ".venv/*" "__pycache__/*" "*/__pycache__/*" ".git/*" ".gitignore" ".DS_Store" "*/.DS_Store" \
  -x "*.ipynb" "make_zip.sh" "build/shrink_index.py" \
  -x "outputs/*" "outputs/" "evaluation/*" "evaluation/" "docs/*" "docs/" \
  -x "data/index/backup/*" "data/chunks_original.jsonl" "data/pdf_fixed.jsonl" "data/pdf_need_ocr.txt"
echo "---- outputs/code.zip ----"
unzip -l outputs/code.zip | tail -n 1
ls -lh outputs/code.zip | awk '{print "サイズ:", $5}'
SIZE=$(stat -f%z outputs/code.zip 2>/dev/null || stat -c%s outputs/code.zip)
if [ "$SIZE" -gt 31457280 ]; then
  echo "⚠ 30MB を超えています。python -m build.shrink_index で索引を半分にしてから作り直してください。"
fi
