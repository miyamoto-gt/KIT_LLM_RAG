"""RAG を段階ごとに診断する。どこで点を落としているかを1回でまとめて見る。

  ① データ   : 正解の根拠が索引の中に1つでもあるか（RAGデータが足りているか）
  ② 検索     : その根拠を、プロンプトに入れる上位 top_k 件の中に拾えているか（何位か）
  ③ 生成     : 資料を見て正しく答えられるか
               ・実際の検索結果を渡したとき
               ・正解の根拠チャンクだけを渡したとき（オラクル）＝検索が完璧なら解けるか
  ④ 計算ツール: 振り分け → <calc>タグ → 式が計算できる → 計算結果が正しい → 最終回答が正しい
  ＋ 確信度   : 生成した答えのトークン確率（平均・最小）と、正解率の関係

使い方:
  （KIT_LLM_RAG の直下から python -m で実行する）
  python -m evaluation.stage_check                     # 全段階（生成するので時間がかかる）
  python -m evaluation.stage_check --stages 1 2        # データと検索だけ（生成なし・速い）
  python -m evaluation.stage_check --ids E01 E02 --stages 4
  python -m evaluation.stage_check --set evaluation/sets/add_question.json

根拠のキーワード:
  既定では evaluation/sets/eval_set.json の expect を使う。ただし「いいえ」「ありません」のような答えや、
  計算結果（1715000 など）は資料にそのまま書かれていないので、①②の判定ができない。
  その問題には evidence を足すと①②も測れる（無ければ「判定不可」として数えない）。
    {"id": "F08", ..., "evidence": [["電子情報システム工学科"], ["工学部"]]}
    {"id": "E01", ..., "evidence": [["入学金"], ["授業料"]]}

結果: 画面に段階ごとの集計、outputs/stage_report.json に1問ずつの詳細。
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

import torch

from config import CFG
from evaluation.evaluate import free_memory, judge, norm
from pipeline import STOP_MARKERS, RAGPipeline, calc_final, ensure_calc_values, postprocess
from prompts import build_calc_answer_prompt, build_prompt
from router import plan
from tools import run_calc_all

YES_NO = {norm(w) for w in ("いいえ", "はい", "ありません", "ない", "存在しません", "違い", "異なり",
                            "別", "ではありません", "ではなく")}
SEARCH_DEPTH = 20     # ② で何位まで探すか
OUT_PATH = "outputs/stage_report.json"


# ---------- 共通 ----------
def groups(words):
    return [[norm(w) for w in g] for g in words]


def contains_all(ntext, ngroups):
    return all(any(w in ntext for w in g) for g in ngroups)


def evidence_of(q):
    """①②で使う根拠キーワード。答えそのもの（はい/いいえ・計算結果）しか無ければ None。"""
    if q.get("evidence"):
        return groups(q["evidence"])
    if not q.get("expect") or q["category"].startswith("E"):
        return None
    g = [grp for grp in groups(q["expect"]) if not set(grp) <= YES_NO]
    return g or None


@torch.no_grad()
def generate_with_conf(gen, prompt, max_new, qtype):
    """生成して、答えの部分（1行目／説明は段落まで）のトークン確率の平均と最小を返す。"""
    if gen.use_chat:
        ids = gen.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                          add_generation_prompt=True, return_tensors="pt")
    else:
        ids = gen.tok(prompt, return_tensors="pt").input_ids
    ids = ids.to(gen.device)
    out = gen.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                             do_sample=False, repetition_penalty=1.1,
                             pad_token_id=gen.tok.pad_token_id or gen.tok.eos_token_id,
                             output_scores=True, return_dict_in_generate=True)
    seq = out.sequences[0, ids.shape[1]:]
    logp = gen.model.compute_transition_scores(out.sequences, out.scores,
                                               normalize_logits=True)[0].float().cpu().tolist()
    raw = gen.tok.decode(seq, skip_special_tokens=True)

    # 答えの部分だけで確信度を測る（後ろの暴走した続きは含めない）
    stops = STOP_MARKERS + (("\n",) if qtype in ("fact", "calc") else ("\n\n",))
    n = len(logp)
    for i in range(1, len(seq) + 1):
        t = gen.tok.decode(seq[:i], skip_special_tokens=True).lstrip()
        if t and any(s in t for s in stops):
            n = max(i - 1, 1)
            break
    lp = [x for x in logp[:n] if math.isfinite(x)] or [0.0]
    return raw, round(math.exp(sum(lp) / len(lp)), 3), round(math.exp(min(lp)), 3)


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


# ---------- 本体 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="evaluation/sets/eval_set.json")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--stages", nargs="*", type=int, default=[1, 2, 3, 4])
    # 検索設定を一時的に変えて比べる（config.py は書き換えない）
    ap.add_argument("--decay", type=float, help="古さによる減点（1.0で無効。既定は config の値）")
    ap.add_argument("--per-source", type=int, help="同じページから取るチャンク数（既定は config の値）")
    ap.add_argument("--top-k", type=int, help="プロンプトに入れる資料の数（既定は config の値）")
    args = ap.parse_args()
    S = set(args.stages)
    import index as _index
    if args.decay is not None:
        _index.RECENCY_DECAY = args.decay
    if args.per_source is not None:
        _index.MAX_PER_SOURCE = args.per_source
    if args.top_k is not None:
        CFG.top_k = args.top_k
    print(f"設定: top_k={CFG.top_k} 古さの減点={_index.RECENCY_DECAY} 同じページ上限={_index.MAX_PER_SOURCE}")

    items = json.load(open(args.set, encoding="utf-8"))
    if args.ids:
        items = [q for q in items if q["id"] in args.ids]

    rag = RAGPipeline() if S & {3, 4} else None
    if rag:
        ret = rag.retriever
    else:
        from index import Retriever
        ret = Retriever()
    print("索引のチャンクを正規化中…", flush=True)
    ntexts = [norm(c.text) for c in ret.chunks]
    K = CFG.top_k

    report = []
    for q in items:
        qid, question = q["id"], q["question"]
        ev = evidence_of(q)
        p = plan(question)
        qtype = p["qtype"]
        r = {"id": qid, "category": q["category"], "question": question, "qtype": qtype, "parts": p["parts"]}

        # ① データ：根拠が1つのチャンクにまとまっているか／ばらばらにあるか／無いか
        gold = None
        if ev is None:
            r["data"] = "判定不可"
        else:
            full = [i for i, t in enumerate(ntexts) if contains_all(t, ev)]
            if full:
                r["data"], gold = "あり", full
            elif all(any(any(w in t for w in g) for t in ntexts) for g in ev):
                r["data"] = "分散"           # 根拠が複数のチャンクに分かれている
            else:
                r["data"] = "なし"
            r["gold_n"] = len(full)

        # ② 検索：上位何位に根拠が出るか、上位K件の資料全体に根拠がそろうか
        hits = ret.search(question, k=max(K, SEARCH_DEPTH)) if S & {2, 3, 4} else []
        top = hits[:K]
        if 2 in S and ev is not None:
            r["rank"] = next((i + 1 for i, (c, _) in enumerate(hits)
                              if contains_all(norm(c.text), ev)), None)
            r["ctx_ok"] = contains_all(norm(" ".join(c.text for c, _ in top)), ev)
            r["top_score"] = round(hits[0][1], 3) if hits else None
            r["top_sources"] = [c.source for c, _ in top]

        # ③ 生成（計算以外）
        if 3 in S and qtype != "calc" and q.get("expect"):
            mx = p["max_new"]
            raw, cm, cmin = generate_with_conf(rag.generate, build_prompt(question, top, qtype), mx, qtype)
            ans = postprocess(raw, qtype, p["max_sent"])
            r.update(answer=ans, gen=judge(ans, q["expect"], q.get("forbid", []))[0],
                     conf=cm, conf_min=cmin)
            if gold:                         # オラクル：正解の根拠チャンクだけを渡す
                oc = ret._expand(gold[0])
                raw_o, _, _ = generate_with_conf(rag.generate, build_prompt(question, [(oc, 1.0)], qtype),
                                                 mx, qtype)
                ans_o = postprocess(raw_o, qtype, p["max_sent"])
                r.update(oracle_answer=ans_o, oracle=judge(ans_o, q["expect"], q.get("forbid", []))[0])
            free_memory()

        # ④ 計算ツール
        if 4 in S and (qtype == "calc" or q["category"].startswith("E")) and q.get("expect"):
            r["routed_calc"] = qtype == "calc"
            raw1, cm1, _ = generate_with_conf(rag.generate, build_prompt(question, top, "calc"),
                                              CFG.max_new_tokens["calc"], "calc")
            r["calc_raw"] = raw1.strip().split("\n")[0][:120]
            r["calc_tag"] = "<calc>" in raw1
            calcs = run_calc_all(raw1, p["n_calc"])
            r["calc_parsed"] = bool(calcs)
            r["calc_conf"] = cm1
            exp_words = {w for g in groups(q["expect"]) for w in g}
            if calcs:
                r["calc_expr"] = " / ".join(e for e, _ in calcs)
                r["calc_value"] = " / ".join(str(v) for _, v in calcs)
                r["calc_value_ok"] = any(norm(str(v)) in exp_words for _, v in calcs)
                prompt, max_new, post_type, post_sent = calc_final(question, top, calcs, p)
                raw2, cm2, cmin2 = generate_with_conf(rag.generate, prompt, max_new, post_type)
                ans = ensure_calc_values(postprocess(raw2, post_type, post_sent), calcs)
            else:
                r["calc_value_ok"] = False
                raw2, cm2, cmin2 = generate_with_conf(rag.generate, build_prompt(question, top, "fact"),
                                                      CFG.max_new_tokens["fact"], "fact")
                ans = postprocess(raw2, "fact")
            r.update(answer=ans, gen=judge(ans, q["expect"], q.get("forbid", []))[0],
                     conf=cm2, conf_min=cmin2)
            free_memory()

        report.append(r)
        print(f"{qid} data={r.get('data')} rank={r.get('rank', '-')} gen={r.get('gen', '-')} "
              f"oracle={r.get('oracle', '-')} conf={r.get('conf', '-')}", flush=True)

    summarize(report, S, K)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(report, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n1問ずつの詳細: {OUT_PATH}")


def summarize(report, S, K):
    print("\n" + "=" * 50)
    judged = [r for r in report if r.get("data") not in (None, "判定不可")]

    if 1 in S:
        c = Counter(r["data"] for r in report)
        print(f"\n【①データ】根拠が索引にあるか（判定できた {len(judged)}問）")
        for k in ("あり", "分散", "なし"):
            print(f"  {k}: {pct(c[k], len(judged))}")
        print(f"  判定不可（evidence 未設定）: {c['判定不可']}問")
        for r in report:
            if r["data"] == "なし":
                print(f"    なし → {r['id']} {r['question']}")

    if 2 in S:
        have = [r for r in judged if r["data"] == "あり"]
        ranks = [r.get("rank") for r in have]
        print(f"\n【②検索】根拠が索引にある {len(have)}問のうち")
        for k in (1, 2, K, 10, SEARCH_DEPTH):
            print(f"  上位{k:>2}件に入った: {pct(sum(1 for x in ranks if x and x <= k), len(have))}")
        mrr = sum(1 / x for x in ranks if x) / len(have) if have else 0
        print(f"  MRR: {mrr:.3f}")
        print(f"  プロンプトの資料（上位{K}件）に根拠がそろった（分散も含む）: "
              f"{pct(sum(1 for r in judged if r.get('ctx_ok')), len(judged))}")
        for r in have:
            if not r.get("rank") or r["rank"] > K:
                print(f"    取れていない → {r['id']} 順位={r.get('rank') or f'{SEARCH_DEPTH}位外'} {r['question']}")
                for s_ in r.get("top_sources", []):
                    print(f"        取った資料: {s_}")

    gen = [r for r in report if r.get("gen") in ("pass", "fail") and "calc_tag" not in r]
    if 3 in S and gen:
        print(f"\n【③生成】（計算以外 {len(gen)}問）")
        print(f"  全体の正解率: {pct(sum(r['gen'] == 'pass' for r in gen), len(gen))}")
        got = [r for r in gen if r.get("ctx_ok")]
        miss = [r for r in gen if r.get("ctx_ok") is False]
        print(f"  資料に根拠があったとき: {pct(sum(r['gen'] == 'pass' for r in got), len(got))}  ← 生成の力")
        print(f"  資料に根拠が無かったとき: {pct(sum(r['gen'] == 'pass' for r in miss), len(miss))}")
        orc = [r for r in gen if r.get("oracle") in ("pass", "fail")]
        print(f"  オラクル（根拠チャンクだけを渡す）: {pct(sum(r['oracle'] == 'pass' for r in orc), len(orc))}"
              "  ← 検索が完璧なら届く上限")
        for r in got:
            if r["gen"] == "fail":
                print(f"    資料はあるのに不正解 → {r['id']} {r['question']} → {r['answer']}")

    calc = [r for r in report if "calc_tag" in r]
    if 4 in S and calc:
        n = len(calc)
        print(f"\n【④計算ツール】（{n}問）")
        print(f"  計算として振り分けられた: {pct(sum(r['routed_calc'] for r in calc), n)}")
        print(f"  <calc>タグを書いた    : {pct(sum(r['calc_tag'] for r in calc), n)}")
        print(f"  式として計算できた    : {pct(sum(r['calc_parsed'] for r in calc), n)}")
        print(f"  計算結果が正しい      : {pct(sum(r['calc_value_ok'] for r in calc), n)}")
        print(f"  最終回答が正しい      : {pct(sum(r['gen'] == 'pass' for r in calc), n)}")
        for r in calc:
            if not r["calc_value_ok"]:
                print(f"    {r['id']} 式={r.get('calc_expr')} 値={r.get('calc_value')} 出力={r['calc_raw']!r}")

    scored = [r for r in report if r.get("gen") in ("pass", "fail") and r.get("conf") is not None]
    if scored:
        print("\n【確信度】答えのトークン確率（平均）と正解率")
        ok = [r["conf"] for r in scored if r["gen"] == "pass"]
        ng = [r["conf"] for r in scored if r["gen"] == "fail"]
        if ok:
            print(f"  正解の平均確信度: {sum(ok) / len(ok):.3f}（{len(ok)}問）")
        if ng:
            print(f"  不正解の平均確信度: {sum(ng) / len(ng):.3f}（{len(ng)}問）")
        bins = defaultdict(lambda: [0, 0])
        edges = (0.5, 0.7, 0.85, 0.95, 1.01)
        for r in scored:
            b = next(e for e in edges if r["conf"] < e)
            bins[b][1] += 1
            bins[b][0] += r["gen"] == "pass"
        lo = 0.0
        for e in edges:
            if bins[e][1]:
                print(f"  {lo:.2f}〜{min(e, 1.0):.2f}: {pct(*bins[e])}")
            lo = e


if __name__ == "__main__":
    main()