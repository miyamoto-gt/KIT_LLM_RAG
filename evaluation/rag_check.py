"""RAG の4段階を、それぞれ独立に測るチェックコード。

  ① データ量 : 巡回・手作りで集めたデータは足りているか
               - 索引全体：チャンク数・ページ数・種類（web/pdf/map/manual）・サイトの区画ごとの量・
                 文字数・年の分布・重複・文字化け
               - 質問ごと：答えの根拠を含むチャンクが索引に何件あるか（0件＝データ不足）
  ② 候補を拾う : 根拠チャンクを、検索の候補（ベクトル検索・キーワード検索 BM25 の上位 N 件）に拾えているか
               - ベクトルだけ／BM25だけ／どちらか で別々に数える（どちらの検索が効いているか分かる）
  ③ 選ぶ      : 候補の中から、最終的にプロンプトに入れる top_k 件に根拠を選べているか
               - 候補にはあったのに top_k から落ちた（＝並べ替え・古さの減点・同じページ上限で落ちた）を数える
               - top_k の中で根拠を含むチャンクの割合（適合率）と、根拠の順位
  ④ 生成      : 選んだ資料をもとに正しく答えられるか
               - 根拠が資料にあったとき／無かったときの正解率、根拠チャンクだけを渡したとき（オラクル）

使い方（KIT_LLM_RAG の直下から）:
  python -m evaluation.rag_check --stages data                 # ①だけ（モデル不要・速い）
  python -m evaluation.rag_check --stages data pick select     # ①〜③（埋め込みモデルだけ使う）
  python -m evaluation.rag_check                                # ①〜④（生成あり・時間がかかる）
  python -m evaluation.rag_check --set evaluation/sets/paraphrase_set.json --pool 50
  python -m evaluation.rag_check --decay 1.0 --per-source 2 --top-k 5   # 検索設定を変えて比べる

根拠キーワードは stage_check と同じく evidence（無ければ expect）を使う。
結果: 画面に段階ごとの集計、outputs/rag_check_<セット名>.json に1問ずつの詳細。
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from urllib.parse import urlparse

import numpy as np

from config import CFG
from evaluation.evaluate import free_memory, judge, norm
from evaluation.stage_check import contains_all, evidence_of, generate_with_conf, pct


# ---------- ① データ量 ----------
def section_of(source: str) -> str:
    """URL をサイトの区画（ドメイン＋最初のパス）にまとめる。手作りデータはそのまま。"""
    if not source.startswith("http"):
        return f"(手作り) {source[:20]}"
    u = urlparse(source)
    first = [p for p in u.path.split("/") if p][:1]
    return f"{u.netloc.replace('www.', '')}/{first[0] if first else ''}"


def readable_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    ok = sum(1 for c in chars if "぀" <= c <= "ヿ" or "一" <= c <= "鿿" or "　" <= c <= "〿"
             or "＀" <= c <= "￯" or " " <= c <= "~")
    return ok / len(chars)


def data_report(chunks, chunk_year) -> dict:
    n = len(chunks)
    lens = np.array([len(c.text) for c in chunks])
    sources = Counter(c.source for c in chunks)
    types = Counter(c.type for c in chunks)
    sections = Counter(section_of(c.source) for c in chunks)
    years = Counter(chunk_year(c) for c in chunks)
    dup = n - len({c.text for c in chunks})
    garbled = sum(readable_ratio(c.text) < 0.85 for c in chunks)

    print("\n【①データ量】索引全体")
    print(f"  チャンク数: {n}   ページ（出典）数: {len(sources)}   重複チャンク: {dup}   文字化けの疑い: {garbled}")
    print(f"  文字数: 平均 {lens.mean():.0f} / 中央値 {np.median(lens):.0f} / 最小 {lens.min()} / 最大 {lens.max()}")
    print("  種類: " + "  ".join(f"{k}={v}" for k, v in types.most_common()))
    print("  サイトの区画（チャンク数の多い順・上位20）:")
    for k, v in sections.most_common(20):
        print(f"    {v:6d}  {k}")
    print("  チャンクの年（推定）: " + "  ".join(
        f"{('不明' if y is None else y)}={v}" for y, v in sorted(years.items(), key=lambda x: (x[0] is None, x[0] or 0))[-12:]))
    big = sources.most_common(5)
    print("  チャンクの多いページ（上位5）: " + " / ".join(f"{s[-45:]}={v}" for s, v in big))
    idx_dir = CFG.index_dir
    for f in ("faiss.index", "chunks.jsonl"):
        p = os.path.join(idx_dir, f)
        if os.path.exists(p):
            print(f"  {p}: {os.path.getsize(p) / 1024 / 1024:.1f}MB")
    return {"chunks": n, "pages": len(sources), "duplicates": dup, "garbled": garbled,
            "types": dict(types), "sections": dict(sections.most_common(50))}


# ---------- ② 候補を拾う / ③ 選ぶ ----------
def candidates(ret, question: str, pool: int):
    """search() の中と同じ方法で、ベクトル検索と BM25 の候補（チャンク番号）をそれぞれ返す。"""
    q = ret.embedder.query(question)
    _, ids = ret.index.search(q, pool)
    dense = [int(i) for i in ids[0] if i >= 0]
    sparse = []
    if getattr(ret, "bm25", None) is not None:
        from index import char_bigrams
        scores = ret.bm25.get_scores(char_bigrams(question))
        sparse = [int(i) for i in np.argsort(-scores)[:pool]]
    return dense, sparse


def first_rank(id_list, gold: set):
    return next((r + 1 for r, i in enumerate(id_list) if i in gold), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="evaluation/sets/public_set.json")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--stages", nargs="*", default=["data", "pick", "select", "gen"],
                    choices=["data", "pick", "select", "gen"])
    ap.add_argument("--pool", type=int, default=40, help="②で見る候補の数（ベクトル・BM25 それぞれ）")
    ap.add_argument("--decay", type=float)
    ap.add_argument("--per-source", type=int)
    ap.add_argument("--top-k", type=int)
    args = ap.parse_args()
    S = set(args.stages)

    import index as _index
    if args.decay is not None:
        _index.RECENCY_DECAY = args.decay
    if args.per_source is not None:
        _index.MAX_PER_SOURCE = args.per_source
    if args.top_k is not None:
        CFG.top_k = args.top_k
    K = CFG.top_k
    print(f"設定: top_k={K} 候補数={args.pool} 古さの減点={_index.RECENCY_DECAY} 同じページ上限={_index.MAX_PER_SOURCE}")

    # 生成するときは RAGPipeline（検索も含む）、しないときは Retriever だけ読み込む
    rag = None
    if "gen" in S:
        from pipeline import RAGPipeline
        rag = RAGPipeline()
        ret = rag.retriever
    elif S & {"pick", "select"}:
        ret = _index.Retriever()
    else:                                      # ①だけならモデルを読み込まない
        from schema import load_chunks
        ret = type("R", (), {})()
        ret.chunks = load_chunks(os.path.join(CFG.index_dir, "chunks.jsonl"))

    items = json.load(open(args.set, encoding="utf-8"))
    if args.ids:
        items = [q for q in items if q["id"] in args.ids]

    out = {"settings": {"top_k": K, "pool": args.pool, "decay": _index.RECENCY_DECAY,
                        "per_source": _index.MAX_PER_SOURCE}, "questions": []}
    if "data" in S:
        out["data"] = data_report(ret.chunks, _index.chunk_year)

    print("\n根拠チャンクを探しています…", flush=True)
    ntexts = [norm(c.text) for c in ret.chunks]

    for q in items:
        ev = evidence_of(q)
        r = {"id": q["id"], "question": q["question"]}
        if ev is None:
            r["data"] = "判定不可"
            out["questions"].append(r)
            continue
        gold = {i for i, t in enumerate(ntexts) if contains_all(t, ev)}
        r["gold_n"] = len(gold)
        r["gold_sections"] = sorted({section_of(ret.chunks[i].source) for i in gold})[:5]
        r["data"] = "あり" if gold else "なし"

        if S & {"pick", "select", "gen"}:
            dense, sparse = candidates(ret, q["question"], args.pool)
            r["dense_rank"] = first_rank(dense, gold)
            r["bm25_rank"] = first_rank(sparse, gold)
            r["picked"] = bool(gold & (set(dense) | set(sparse)))

            hits = ret.search(q["question"], k=K)
            texts = [norm(c.text) for c, _ in hits]
            rel = [contains_all(t, ev) for t in texts]
            part = [sum(any(w in t for w in g) for g in ev) / len(ev) for t in texts]
            r["selected_rank"] = next((i + 1 for i, x in enumerate(rel) if x), None)
            r["selected"] = any(rel)
            r["ctx_ok"] = contains_all(" ".join(texts), ev)
            r["precision"] = round(sum(rel) / len(rel), 2) if rel else 0.0
            r["partial_precision"] = round(sum(part) / len(part), 2) if part else 0.0
            r["top_sources"] = [c.source for c, _ in hits]

        if "gen" in S and q.get("expect"):
            from pipeline import calc_final, ensure_calc_values, postprocess
            from prompts import build_prompt
            from router import plan
            from tools import run_calc_all
            p = plan(q["question"])
            raw, conf, _ = generate_with_conf(rag.generate, build_prompt(q["question"], hits, p["qtype"]),
                                              p["max_new"], p["qtype"])
            if p["qtype"] == "calc":                 # 本番（pipeline.answer）と同じ流れ：式を計算 → 結果を添えて答え直す
                calcs = run_calc_all(raw, p["n_calc"])
                if calcs:
                    prompt, mx, post_type, post_sent = calc_final(q["question"], hits, calcs, p)
                    raw, conf, _ = generate_with_conf(rag.generate, prompt, mx, post_type)
                    ans = ensure_calc_values(postprocess(raw, post_type, post_sent), calcs)
                else:
                    raw, conf, _ = generate_with_conf(rag.generate, build_prompt(q["question"], hits, "fact"),
                                                      CFG.max_new_tokens["fact"], "fact")
                    ans = postprocess(raw, "fact")
            else:
                ans = postprocess(raw, p["qtype"], p["max_sent"])
            r.update(qtype=p["qtype"], answer=ans, conf=conf,
                     gen=judge(ans, q["expect"], q.get("forbid", []))[0])
            if gold and p["qtype"] != "calc":
                oc = ret._expand(sorted(gold)[0])
                raw_o, _, _ = generate_with_conf(rag.generate, build_prompt(q["question"], [(oc, 1.0)], p["qtype"]),
                                                 p["max_new"], p["qtype"])
                r["oracle"] = judge(postprocess(raw_o, p["qtype"], p["max_sent"]), q["expect"], q.get("forbid", []))[0]
            free_memory()

        out["questions"].append(r)
        print(f"{q['id']} 根拠={r.get('gold_n', '-')}件 ベクトル={r.get('dense_rank', '-')}位 "
              f"BM25={r.get('bm25_rank', '-')}位 選択={r.get('selected_rank', '-')}位 生成={r.get('gen', '-')}",
              flush=True)

    summarize(out["questions"], S, K, args.pool)
    name = os.path.splitext(os.path.basename(args.set))[0]
    os.makedirs("outputs", exist_ok=True)
    path = f"outputs/rag_check_{name}.json"
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n1問ずつの詳細: {path}")


def summarize(rows, S, K, pool):
    judged = [r for r in rows if r.get("data") in ("あり", "なし")]
    have = [r for r in judged if r["data"] == "あり"]
    print("\n" + "=" * 60)
    print(f"\n【①データ量】質問ごと（判定できた {len(judged)}問）")
    print(f"  根拠が索引にある: {pct(len(have), len(judged))}")
    gn = sorted(r["gold_n"] for r in have)
    if gn:
        print(f"  根拠チャンクの件数: 中央値 {gn[len(gn) // 2]} / 最小 {gn[0]} / 最大 {gn[-1]}"
              f"（1件だけの質問 {sum(x == 1 for x in gn)}問 ← 取りこぼすと答えられない）")
    for r in judged:
        if r["data"] == "なし":
            print(f"    データ不足 → {r['id']} {r['question']}")

    if S & {"pick", "select", "gen"} and have:
        n = len(have)
        d = sum(1 for r in have if r.get("dense_rank"))
        b = sum(1 for r in have if r.get("bm25_rank"))
        u = sum(1 for r in have if r.get("picked"))
        print(f"\n【②候補を拾う】根拠がある {n}問のうち、候補（各{pool}件）に根拠が入った")
        print(f"  ベクトル検索だけ: {pct(d, n)}   BM25だけ: {pct(b, n)}   どちらか: {pct(u, n)}")
        only_d = sum(1 for r in have if r.get("dense_rank") and not r.get("bm25_rank"))
        only_b = sum(1 for r in have if r.get("bm25_rank") and not r.get("dense_rank"))
        print(f"  片方でしか拾えなかった: ベクトルだけ {only_d}問 / BM25だけ {only_b}問")
        for r in have:
            if not r.get("picked"):
                print(f"    候補に無い → {r['id']} {r['question']}")

        sel = [r for r in have if r.get("picked")]
        ok = sum(1 for r in sel if r.get("selected"))
        print(f"\n【③選ぶ】候補に根拠があった {len(sel)}問のうち、top_{K} に選べた: {pct(ok, len(sel))}")
        ranks = [r["selected_rank"] for r in have if r.get("selected_rank")]
        if ranks:
            c = Counter(ranks)
            print("  根拠の順位: " + "  ".join(f"{k}位={c[k]}" for k in sorted(c)))
        print(f"  top_{K} の適合率（根拠を含むチャンクの割合）: 平均 "
              f"{np.mean([r['precision'] for r in have]):.2f}"
              f"   部分一致で数えると {np.mean([r['partial_precision'] for r in have]):.2f}")
        print(f"  資料全体に根拠がそろった（複数チャンクに分かれていても可）: "
              f"{pct(sum(1 for r in have if r.get('ctx_ok')), n)}")
        for r in sel:
            if not r.get("selected"):
                print(f"    候補にはあったのに落ちた → {r['id']} ベクトル{r.get('dense_rank')}位 "
                      f"BM25{r.get('bm25_rank')}位 {r['question']}")

    gen = [r for r in rows if r.get("gen") in ("pass", "fail")]
    if "gen" in S and gen:
        print(f"\n【④生成】（{len(gen)}問）")
        print(f"  全体の正解率: {pct(sum(r['gen'] == 'pass' for r in gen), len(gen))}")
        got = [r for r in gen if r.get("ctx_ok")]
        miss = [r for r in gen if r.get("ctx_ok") is False]
        print(f"  資料に根拠があったとき: {pct(sum(r['gen'] == 'pass' for r in got), len(got))}  ← 生成の力")
        print(f"  資料に根拠が無かったとき: {pct(sum(r['gen'] == 'pass' for r in miss), len(miss))}")
        orc = [r for r in gen if r.get("oracle") in ("pass", "fail")]
        print(f"  オラクル（根拠チャンク1件だけ）: {pct(sum(r['oracle'] == 'pass' for r in orc), len(orc))}"
              "  ※根拠が2チャンクにまたがる質問では低く出る")
        # どの段階で落ちたかの内訳
        where = Counter()
        for r in gen:
            if r["gen"] == "pass":
                where["正解"] += 1
            elif r.get("data") == "なし":
                where["①データ不足"] += 1
            elif not r.get("picked"):
                where["②候補に無い"] += 1
            elif not r.get("ctx_ok"):
                where["③選べていない"] += 1
            else:
                where["④生成の誤り"] += 1
        print("  落ちた段階の内訳: " + "  ".join(f"{k}={v}" for k, v in where.most_common()))


if __name__ == "__main__":
    main()