"""API 版の診断：検索 → 渡した資料 → 生成 を段階ごとに測り、「正解の資料だけ」を渡した上限（oracle）と比べる。

提出用の submission/kitrag をそのまま使う（本番と同じ検索・プロンプト・後処理）。
API の URL とトークンは `python submit.py login` で保存された ~/.kitrag.json から読む。

使い方（KIT_LLM_RAG の直下から）:
  python evaluation/api_check.py --set evaluation/sets/my_new69.json
  python evaluation/api_check.py --set evaluation/sets/public_set.json --mode normal
  python evaluation/api_check.py --set ... --top-k 2 --narrow-keep 4 --tag k2      # 渡す量を変えて比べる
  python evaluation/api_check.py --set ... --embed-model cl-nagoya/ruri-v3-310m --mode retrieval  # 埋め込みの比較

段階:
  ① 候補   : 正解の根拠を含むチャンクが、埋め込み／BM25 の候補（上位40件）に入ったか
  ② 選ぶ   : 生成に渡す上位 top_k 件に根拠が入ったか
  ③ 渡す   : 絞り込み（narrow）のあとの資料に根拠が残ったか
  ④ 生成   : 普段どおり答えて正解か
  ⑤ oracle : 根拠を含むチャンク1件だけを渡して答えて正解か（検索が完璧なときの上限）
失敗の分類（論文 arXiv:2603.11513 の分類を参考）:
  拒否（「記載されていません」など）／一部だけ（正解キーワードの一部）／資料に無い内容（資料との重なりが小さい）／別の値
--mode : both（④⑤、既定）・normal（④だけ）・oracle（⑤だけ）・retrieval（①〜③だけ、生成なし）
結果  : 画面に集計、outputs/api_check_<セット名>[_<tag>].json に1問ずつの詳細
API の使用回数の目安：1問あたり 埋め込み1回＋生成1〜2回（both なら2〜4回）。1日5000回まで。
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------- 判定（evaluation/evaluate.py と同じ規則。torch を読まないようにここに写す）----------
def _kanji_amount(s):
    def man(m):
        a = int(m.group(1)) * 10000
        rest = m.group(2) or ""
        if rest.endswith("千"):
            a += int(rest[:-1]) * 1000
        elif rest:
            a += int(rest)
        return str(a)
    s = re.sub(r"(\d+)万(\d+千|\d{1,4}(?!\d))?", man, s)
    return re.sub(r"(\d+)千", lambda m: str(int(m.group(1)) * 1000), s)


def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    for ch in (" ", "　", ",", "，", "円", "\n"):
        s = s.replace(ch, "")
    return _kanji_amount(s)


def groups(words):
    return [[norm(w) for w in g] for g in words]


def contains_all(ntext, ngroups):
    return all(any(w in ntext for w in g) for g in ngroups)


def judge(answer, expect, forbid=()):
    a = norm(answer)
    if not expect:
        return "skip", "正解未設定"
    if any(norm(w) in a for w in forbid):
        return "fail", "禁止語"
    missing = [g for g in expect if not any(norm(w) in a for w in g)]
    return ("fail", f"不足: {[g[0] for g in missing]}") if missing else ("pass", "")


YES_NO = {norm(w) for w in ("いいえ", "はい", "ありません", "ない", "存在しません", "違い", "異なり", "別",
                            "ではありません", "ではなく")}


def evidence_of(q):
    """根拠キーワード（evidence。無ければ expect。計算問題・はい/いいえだけなら None）。"""
    if q.get("evidence"):
        return groups(q["evidence"])
    if not q.get("expect") or q.get("category", "").startswith("E"):
        return None
    g = [grp for grp in groups(q["expect"]) if not set(grp) <= YES_NO]
    return g or None


# ---------- 失敗の分類 ----------
_REFUSAL = re.compile(r"記載されていません|書かれていません|見当たりません|わかりません|分かりません|"
                      r"できません|特定でき|情報はありません|含まれていません|^不明")


def bigrams(s):
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def classify(answer, expect, ctx):
    if _REFUSAL.search(answer or ""):
        return "拒否"
    a = norm(answer)
    hit = sum(any(norm(w) in a for w in g) for g in expect)
    if 0 < hit < len(expect):
        return "一部だけ"
    ab = bigrams(answer)
    overlap = len(ab & bigrams(ctx)) / max(len(ab), 1)
    if overlap < 0.4:
        return "資料に無い内容"
    return "別の値"


# ---------- 準備 ----------
def setup_env(args):
    cfg_path = Path.home() / ".kitrag.json"
    if not os.environ.get("LLM_BASE_URL"):
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            sys.exit("先に `python submit.py login --url <サーバーURL>` を実行してください。")
        url = cfg["url"].rstrip("/") + "/api/llm/v1"
        os.environ.update({"LLM_BASE_URL": url, "OPENAI_BASE_URL": url, "OPENAI_API_KEY": cfg["token"]})
    os.environ["KITRAG_TIME_LIMIT"] = str(10 ** 9)          # 評価では締め切りで打ち切らない
    os.environ["KITRAG_NARROW"] = "1" if args.narrow else "0"
    if args.embed_model:
        os.environ["KITRAG_EMBED_MODEL"] = args.embed_model
    sys.path.insert(0, str(Path(args.code).resolve()))


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f}%)" if b else "0/0"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="評価セット（json）")
    ap.add_argument("--code", default="submission", help="main.py と kitrag/ があるフォルダ")
    ap.add_argument("--mode", default="both", choices=["both", "normal", "oracle", "retrieval"])
    ap.add_argument("--top-k", type=int, default=None, help="生成に渡すチャンク数（既定は settings.TOP_K）")
    ap.add_argument("--narrow", type=int, default=1, choices=[0, 1], help="資料を関係する文に絞るか")
    ap.add_argument("--narrow-keep", type=int, default=None, help="絞り込みで残す文の数（既定は settings.NARROW_KEEP）")
    ap.add_argument("--embed-model", default=None, help="埋め込みモデル（例: cl-nagoya/ruri-v3-310m）")
    ap.add_argument("--ids", nargs="*", help="この質問だけ")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="", help="結果ファイル名に付ける印（設定を変えて比べるとき）")
    args = ap.parse_args()
    setup_env(args)

    from kitrag import settings as S
    from kitrag.knowledge import load_chunks
    from kitrag.pipeline import RAG
    from kitrag.prompts import build_context
    from kitrag.router import plan
    if args.top_k:
        S.TOP_K = args.top_k
    if args.narrow_keep:
        S.NARROW_KEEP = args.narrow_keep

    qs = json.load(open(args.set, encoding="utf-8"))
    if args.ids:
        qs = [q for q in qs if q["id"] in set(args.ids)]
    t0 = time.time()
    chunks = load_chunks()
    rag = RAG(chunks, log=lambda m: print(f"  {m}", flush=True))
    R = rag.retriever
    nchunks = [norm(c.text) for c in chunks]
    print(f"資料 {len(chunks)} チャンク・{len(qs)} 問・埋め込み {S.EMBEDDING_MODEL}・top_k {S.TOP_K}・"
          f"絞り込み {'あり（' + str(S.NARROW_KEEP) + '文）' if S.NARROW else 'なし'}（準備 {time.time() - t0:.0f}秒）")

    def one(q):
        ev = evidence_of(q)
        expect = q.get("expect") or []
        r = {"id": q["id"], "question": q["question"], "category": q.get("category", ""), "expect": expect}
        gold = [i for i, t in enumerate(nchunks) if ev and contains_all(t, ev)] if ev else []
        r["gold_n"] = len(gold)
        hits, info = R.search(q["question"], k=S.TOP_K, debug=True)
        ctx = build_context(q["question"], hits, plan(q["question"])["qtype"])
        if ev:
            gs = set(gold)
            r["pool_dense"] = bool(gs & set(info["dense"]))
            r["pool_bm25"] = bool(gs & set(info["bm25"]))
            r["topk"] = any(contains_all(norm(c.text), ev) for c, _ in hits)
            r["ctx"] = contains_all(norm(ctx), ev)
        r["sources"] = [c.source for c, _ in hits]
        if args.mode in ("both", "normal"):
            out = rag.answer(q["question"], hits=hits)
            r["answer"] = out["answer"]
            r["normal"], r["normal_why"] = judge(out["answer"], expect, q.get("forbid", []))
            if r["normal"] == "fail":
                r["normal_class"] = classify(out["answer"], expect, out["context"])
        if args.mode in ("both", "oracle") and gold:
            rank = {i: n for n, i in enumerate(info["bm25"])}
            best = min(gold, key=lambda i: rank.get(i, 10 ** 6))      # 根拠チャンクのうち BM25 で一番上のもの
            out = rag.answer(q["question"], hits=[(R._expand(chunks[best]), 1.0)])
            r["oracle_answer"] = out["answer"]
            r["oracle"], r["oracle_why"] = judge(out["answer"], expect, q.get("forbid", []))
            if r["oracle"] == "fail":
                r["oracle_class"] = classify(out["answer"], expect, out["context"])
        return r

    rows, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(one, qs):
            rows.append(r)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(qs)} 問（{time.time() - t0:.0f}秒）", flush=True)

    # ---------- 集計 ----------
    withev = [r for r in rows if "topk" in r]
    print("\n【①〜③ 検索】根拠キーワードがある問題", len(withev), "問")
    print("  資料に根拠が無い（データ不足） :", pct(sum(r["gold_n"] == 0 for r in withev), len(withev)))
    hasgold = [r for r in withev if r["gold_n"] > 0]
    print("  ① 候補に入った  埋め込み", pct(sum(r["pool_dense"] for r in hasgold), len(hasgold)),
          " BM25", pct(sum(r["pool_bm25"] for r in hasgold), len(hasgold)),
          " どちらか", pct(sum(r["pool_dense"] or r["pool_bm25"] for r in hasgold), len(hasgold)))
    print("  ② 上位", S.TOP_K, "件に入った", pct(sum(r["topk"] for r in hasgold), len(hasgold)))
    print("  ③ 絞った資料に残った", pct(sum(r["ctx"] for r in hasgold), len(hasgold)))

    judged = [r for r in rows if r.get("normal") in ("pass", "fail")]
    if judged:
        print("\n【④ 生成】正解", pct(sum(r["normal"] == "pass" for r in judged), len(judged)))
        g = [r for r in judged if r.get("ctx")]
        print("  渡した資料に根拠があった問題での正解", pct(sum(r["normal"] == "pass" for r in g), len(g)),
              "← ここで落ちるのが「資料はあるのに答えられない」")
        c = Counter(r["normal_class"] for r in judged if r["normal"] == "fail")
        print("  失敗の内訳:", "  ".join(f"{k} {v}" for k, v in c.most_common()) or "なし")
    orc = [r for r in rows if r.get("oracle") in ("pass", "fail")]
    if orc:
        print("\n【⑤ oracle】根拠チャンク1件だけを渡したときの正解", pct(sum(r["oracle"] == "pass" for r in orc), len(orc)))
        c = Counter(r["oracle_class"] for r in orc if r["oracle"] == "fail")
        print("  失敗の内訳:", "  ".join(f"{k} {v}" for k, v in c.most_common()) or "なし")
        both = [r for r in orc if r.get("normal") in ("pass", "fail")]
        if both:
            m = Counter((r["normal"], r["oracle"]) for r in both)
            print(f"  普段 ○ / oracle ○ {m[('pass', 'pass')]}   普段 × / oracle ○ {m[('fail', 'pass')]}（検索・渡し方で直せる）"
                  f"   普段 ○ / oracle × {m[('pass', 'fail')]}   普段 × / oracle × {m[('fail', 'fail')]}（生成そのもの）")
    fails = [r for r in judged if r["normal"] == "fail"]
    if fails:
        print("\n【間違えた問題】")
        for r in fails:
            stage = "データ不足" if r.get("gold_n") == 0 else ("検索" if "ctx" in r and not r["ctx"] else "生成")
            print(f"  {r['id']} [{stage}/{r['normal_class']}] {r['question'][:30]} → {r['answer'][:60]}")

    os.makedirs("outputs", exist_ok=True)
    name = Path(args.set).stem + (f"_{args.tag}" if args.tag else "")
    out = f"outputs/api_check_{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"settings": {"embed": S.EMBEDDING_MODEL, "top_k": S.TOP_K, "narrow": S.NARROW,
                                "narrow_keep": S.NARROW_KEEP, "mode": args.mode}, "rows": rows},
                  f, ensure_ascii=False, indent=1)
    print(f"\n→ {out}（{time.time() - t0:.0f}秒）")


if __name__ == "__main__":
    main()
