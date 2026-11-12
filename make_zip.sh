#!/bin/bash
# 提出用 zip（code.zip、上限30MB）を作る。KIT_LLM_RAG の直下で実行する:  bash make_zip.sh
# 実行時に必要なのは「コード」と「data/index/（chunks.jsonl + faiss.index）」と小さい補助データだけ。
set -e
rm -f code.zip
zip -r -9 code.zip . \
  -x ".venv/*" "__pycache__/*" "*/__pycache__/*" ".git/*" ".gitignore" ".DS_Store" "*/.DS_Store" \
  -x "*.ipynb" "image.png" "code.zip" "make_zip.sh" "shrink_index.py" \
  -x "answers*.json" "answers*.jsonl" "eval_*" "debug_*" "test_submission.csv" "main_test.csv" \
  -x "submission*.csv" "check_mps.py" "compare_eval.py" \
  -x "data/index/backup/*" "data/chunks_original.jsonl" "data/pdf_fixed.jsonl" "data/pdf_need_ocr.txt"
echo "---- code.zip ----"
unzip -l code.zip | tail -n 1
ls -lh code.zip | awk '{print "サイズ:", $5}'
SIZE=$(stat -f%z code.zip 2>/dev/null || stat -c%s code.zip)
if [ "$SIZE" -gt 31457280 ]; then
  echo "⚠ 30MB を超えています。python shrink_index.py で索引を半分にしてから作り直してください。"
fi