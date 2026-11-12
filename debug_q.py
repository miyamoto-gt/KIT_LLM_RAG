"""指定した質問IDについて、検索結果・プロンプト全文・生成結果を表示する。
使い方: python debug_q.py q006
"""
import sys
import csv
import torch

import pipeline
from pipeline import RAGPipeline
from prompts import build_prompt
from config import CFG

qid = sys.argv[1]
with open("questions-2.csv", encoding="utf-8-sig") as f:
    question = next(r["question"] for r in csv.DictReader(f) if r["question_id"] == qid)

p = RAGPipeline()
qtype = pipeline.classify(question)
hits = p.retrieve(question)

print(f"=== {qid} ({qtype}) {question}\n")
print("=== 検索結果")
for i, (chunk, score) in enumerate(hits):
    print(f"[{i}] score={score:.3f} source={getattr(chunk, 'source', '?')}")

prompt = build_prompt(question, hits, qtype)
print("\n=== プロンプト全文")
print(prompt)

print("\n=== 生成 (repetition_penalty=1.1, 現行)")
print(p.generate(prompt, CFG.max_new_tokens[qtype]))

# 比較: repetition_penalty=1.0
g = p.generate
ids = g.tok(prompt, return_tensors="pt").input_ids.to(g.device)
with torch.no_grad():
    out = g.model.generate(ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=CFG.max_new_tokens[qtype], do_sample=False,
                           repetition_penalty=1.0,
                           pad_token_id=g.tok.pad_token_id or g.tok.eos_token_id)
print("\n=== 生成 (repetition_penalty=1.0)")
print(g.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))

def gen(prompt_text):
    return p.generate(prompt_text, CFG.max_new_tokens[qtype]).split("\n")[0]

# A: 年表の各項目を改行で区切る
pA = prompt.replace(" - 19", "\n- 19").replace(" - 20", "\n- 20")
print("\n=== A 改行で区切る:", gen(pA))

# B: 低スコアの資料を除く(0.8未満を落とす)
hitsB = [h for h in hits if h[1] >= 0.8]
print("=== B 低スコア除外:", gen(build_prompt(question, hitsB, qtype)))

# C: 年表を読む few-shot 例を1つ足す
ex = ("資料:\n[1] - 1990年4月1日 A学科設置\n- 1991年4月1日 B学科設置\n"
      "質問: B学科が設置された年はいつですか？\n答え: 1991年です。\n\n")
pC = prompt.replace("資料:\n[1] 【個人", ex + "資料:\n[1] 【個人", 1)
print("=== C 年表の例を追加:", gen(pC))

import re

enk = [h for h in hits if "enkaku" in str(getattr(h[0], "source", ""))]

# D: enkaku だけ + 改行区切り
pD = build_prompt(question, enk, qtype).replace(" - 19", "\n- 19").replace(" - 20", "\n- 20")
print("=== D enkakuのみ+改行:", gen(pD))

# E: D に加えて「項目(日付)」の順に並べ替え
pE = re.sub(r"- (\d{4}年\d+月\d+日)([^\n]+)", r"- \2（\1）", pD)
print("=== E 項目→日付の順:", gen(pE))

# F: few-shotなしの最小プロンプト(enkakuの該当行付近だけ)
i = pD.find("1965年")
F = "資料:\n" + pD[i-2:i+120] + f"\n質問: {question}\n答え:"
print("=== F 最小:", gen(F))
print("--- F の資料部分 ---\n" + pD[i-2:i+120])


r4 = [h for h in hits if "r4tenken" in str(getattr(h[0], "source", ""))]
ks = [h for h in hits if "kessan" in str(getattr(h[0], "source", ""))]

def nl(t):  # 西暦も和暦も、日付の前で改行する
    t = re.sub(r" - (\d{4}年)", r"\n- \1", t)
    return re.sub(r" ?(昭和|平成|令和) ?(\d+) ?年", r"\n\1\2年", t)

print("=== G enkaku+r4:    ", gen(build_prompt(question, enk + r4, qtype)))
print("=== H enkaku+kessan:", gen(build_prompt(question, enk + ks, qtype)))
print("=== I 全部+和暦も改行:", gen(nl(prompt)))


def compress(text, kws, win=1):
    segs = [s for s in nl(text).split("\n") if s.strip()]
    keep = set()
    for i, s in enumerate(segs):
        if any(k in s for k in kws):
            keep.update(range(max(0, i - win), min(len(segs), i + win + 1)))
    return "\n".join(segs[i] for i in sorted(keep))

# 質問から2文字以上の漢字・カタカナの連なりをキーワードにする(簡易版)
kws = [w for w in re.findall(r"[一-龥ァ-ヶー]{2,}", question) if w not in ("設置", "工学部")]
print("kws:", kws)

pJ_ctx = "\n\n".join(compress(c.text, kws) for c, _ in hits if compress(c.text, kws))
J = "資料:\n" + pJ_ctx + f"\n質問: {question}\n答え:"
print("=== J 圧縮:", gen(J))
print("--- J 資料 ---\n" + pJ_ctx)