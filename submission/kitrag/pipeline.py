"""質問1つに答える中心部品。RAG(chunks).answer(question) -> dict。

  振り分け（router.plan）→ 検索（Retriever）→ 資料を絞る（prompts.build_context）
  → 生成（API）→ 計算問題なら式を計算して答え直す → 後処理
LLM が失敗したら、絞った資料の一番関係する文を答えにする（「不明」は0点なので空にしない）。
"""
import re

from . import llm
from . import settings as S
from .postprocess import ensure_calc_values, postprocess
from .prompts import build_calc_answer_messages, build_context, build_extract_messages, build_messages
from .retriever import Retriever
from .router import plan
from .tools import calc_from_values, clean_values, exprs_in_text, parse_values, run_calc_all, values_in_question
from .repair import decompose, drop_refusal_sentences, is_refusal, relax_query, repair_message, shape_missing, unit_missing


_CRUMB = re.compile(r"[（(]?\s*TOP\s*[>＞][^）)。]*[）)]?")             # パンくず「（TOP > 大学案内 > …）」
_NOISE = re.compile(r"https?://|^[>＞#\-・■●※]|^(出典|確認日|種類)\s*[:：]")


def _ext_sentences(ctx: str) -> list:
    """資料を、答えに使える文の候補に分ける（パンくず・URL・見出し・短すぎる切れ端は除く）。"""
    out = []
    for line in ctx.splitlines():
        body = re.sub(r"^\[\d+\]\s*(【[^】]*】\S*\s*)?", "", line).strip()
        body = _CRUMB.sub("", body)
        prev = ""
        for s in re.split(r"(?<=。)|／|\s{2,}", body):
            s = s.strip(" 　")
            if re.fullmatch(r"[（(][^）)]{1,25}[）)]。?", s) and out:   # 「（1965年4月1日）」だけのかけらは、直前の文につける
                out[-1] = ((out[-1][0].rstrip("。") + s)[:200], out[-1][1])
                prev = ""
                continue
            if s[:1] in "：:":                        # 年表で日付と中身が切れたもの
                if prev and not prev.endswith("。") and len(prev) <= 20:   # 直前が日付だけ → つなぎ直す
                    s = prev + s
                    if out and out[-1][0] == prev[:200]:
                        out.pop()
                else:
                    s = s[1:]
            prev = s
            if s.count("|") >= 2:                      # 表の行は「項目、値、…」の形にして候補に残す（点は下げる）
                cells = [x.strip() for x in s.strip("|").split("|") if x.strip()]
                prose = [x for x in cells if len(x) >= 15 and re.search(r"(。|です|ます)$", x)]
                cells = [x for x in cells if x not in prose]   # 表の後ろにつながった普通の文は別の候補
                parts = [("、".join(cells), True)] + [(x, False) for x in prose]
            else:
                parts = [(s, False)]
            for s, table in parts:
                s = re.sub(r"^#+\s*|\*\*", "", s).strip()
                if len(s) < 8 or _NOISE.search(s) or any(s[:200] == o for o, _ in out):
                    continue
                out.append((s[:200], table))
    return out


def _junk(s: str) -> bool:
    """目次（「………」）や、数字と空白ばかりの表の切れ端。"""
    if "……" in s or "・・・・" in s:
        return True
    noise = sum(1 for c in s if c.isdigit() or c in " 　()（）|")
    return noise / max(len(s), 1) > 0.45


_EXT_RULES = [   # 簡易回答だけで使う、答えの形（repair の _UNIT_RULES より先に見る）
    (r"何年間", r"[0-9０-９一二三四五六]+\s*年間"),
    (r"何[%％]|率は", r"[0-9.]+\s*[%％]"),
    (r"開館時間|何時|時間帯|受付時間|窓口.*時間", r"\d{1,2}\s*[:：時]\s*\d{0,2}"),
]


def rank_sentences(ctx: str, question: str) -> list:
    """資料の文を、質問との重なりの点数で並べる → [(点数, 文, 答えの形を含むか)]（高い順）。

    点数 = 質問の文字bigram のうち、その文に含まれるものの重み（資料の中で珍しいほど重い）
         ＋ 質問が求める形（何人→「〇人」、何号館→「〇号館」、いつ→年 など）を含めば加点
    表の行・短い文・数字ばかりの切れ端は下げる。
    """
    import math
    from .prompts import _KANA2, _bigrams
    from .repair import _UNIT_RULES
    cands = [(s, t) for s, t in _ext_sentences(ctx) if "……" not in s]
    if not cands:
        return []
    qtext = re.sub(r"金沢工業大学|KIT|本学", "", question)      # どの資料にも出る大学名は手がかりにしない
    q = {b for b in _bigrams(qtext) if not _KANA2.match(b)}
    n = len(cands)
    df = {b: sum(1 for s, _ in cands if b in s) for b in q}
    w = {b: math.log(1 + n / df[b]) for b in q if df[b]}
    rules = _EXT_RULES + [(qp, a) for qp, a, _ in _UNIT_RULES]
    need = next((a for qp, a in rules if re.search(qp, question)), None)
    top = max(w.values(), default=1.0)
    out = []
    for i, (s, table) in enumerate(cands):
        sc = sum(v for b, v in w.items() if b in s)
        has = bool(need and re.search(need, s))
        if has:
            sc += 2 * top
        if table:
            sc *= 0.8
        if len(s) < 15:
            sc *= 0.7
        if _junk(s):
            sc *= 0.3
        sc -= 0.01 * i                                   # 同点なら前（検索の上位）の文
        out.append((sc, s, has if need else None))
    out.sort(key=lambda x: -x[0])
    return out


def extractive(ctx: str, question: str = "") -> str:
    """LLM が使えないときの答え：資料の中で、質問といちばん重なる文（API は使わない）。

    ・複数を聞く質問（「AとB」「それぞれ」「2つ挙げて」など）は、上位の文を部分の数だけつなぐ
    ・いちばんの文に、質問が求める形（〇人・〇号館・年など）が無ければ、それを含む最上位の文を足す
    ・質問が無ければ従来どおり最初の文
    """
    if not question:
        c = _ext_sentences(ctx)
        return c[0][0] if c else S.FALLBACK
    ranked = rank_sentences(ctx, question)
    if not ranked:
        return S.FALLBACK
    from .router import count_parts, plan
    qtype = plan(question)["qtype"]
    k = min(max(count_parts(question), 1), 3)
    if k == 1 and (qtype != "fact" or ranked[0][2] is not True):
        k = 2                                            # 説明の質問・答えの形が確かめられない質問は2文（取りこぼしを減らす）
    picked = []
    for sc, sent, has in ranked:
        if len(picked) >= k:
            break
        if sc > 0 and sent not in picked:
            picked.append(sent)
    if not picked:
        picked = [ranked[0][1]]
    if ranked[0][2] is False:                            # 答えの形が無い → それを含む文を足す
        extra = next((sent for sc, sent, has in ranked if has and sc > 0 and sent not in picked), None)
        if extra:
            picked.append(extra)
    if qtype == "calc" and re.search(r"合計|合わせ|合算|足し|足す|差|何倍|平均|総額|引", question):   # 計算をはっきり求める質問だけ（「割合は何％」は資料の数をそのまま答える）
        from .tools import calc_from_values, parse_number
        qv = values_in_question(question)                # 質問が数を与えていれば、それで計算して、資料のいちばんの文を添える
        if qv:
            qc = calc_from_values(question, qv)
            if qc:
                expr, val = qc[0]
                unit = re.search(r"[0-9]\s*(万円|円|名|人|件|校)", question)
                unit = unit.group(1) if unit else ("名" if re.search(r"人員|人数|学生数", question) else "")
                head = f"{expr} = {val:,}{unit}です。" if isinstance(val, int) else f"{expr} = {val}{unit}です。"
                return (head + picked[0].rstrip("。") + "。")[:400]
        vals = [(m.group(1), parse_number(m.group(2))) for m in
                re.finditer(r"([^\s、。,，0-9]{1,15}?)\s*([0-9][0-9,，.]*)\s*(?:円|人|名|件|校|%|％)", picked[0])]
        vals = [(n, v) for n, v in vals if v is not None]
        names = [n for n, _ in vals]
        ok = len(set(names)) == len(names) and not any(re.search(r"年度|入学者|月", n) for n in names) \
            and not re.search(r"[0-9]\s*[%％]", picked[0])   # 年度違いの同じ項目・％は足さない
        calcs = calc_from_values(question, vals) if ok and len(vals) >= 2 else []
        if calcs:
            expr, val = calcs[0]
            unit = re.search(r"[0-9]\s*(円|人|名|件|校|%|％)", picked[0]).group(1)
            return f"{expr} = {val:,}{unit}です。" if isinstance(val, int) else f"{expr} = {val}{unit}です。"
    out = "".join(x if x.endswith("。") else x + "。" for x in picked)
    return out[:400]


class RAG:
    def __init__(self, chunks, log=print):
        self.retriever = Retriever(chunks, log=log)
        self.log = log

    def answer(self, question: str, hits=None, split=True) -> dict:
        """hits を渡すと検索をとばしてその資料で答える（評価で「正解の資料だけ」を渡すとき用）。"""
        subs = decompose(question) if (split and hits is None and S.DECOMPOSE) else []
        if subs:                                        # 「AとBは、それぞれ〜」→ 1つずつ検索・生成してつなぐ
            outs = [self.answer(sq, split=False) for sq in subs]
            ans = "".join(o["answer"] if o["answer"].endswith("。") else o["answer"] + "。" for o in outs)
            return {"question": question, "answer": ans[:S.MAX_ANSWER_CHARS], "raw": " / ".join(o["raw"] for o in outs),
                    "qtype": "split", "parts": len(subs), "sources": [s for o in outs for s in o["sources"]],
                    "context": "\n".join(o["context"] for o in outs)}
        p = plan(question)
        qtype = p["qtype"]
        if hits is None:
            hits = self.retriever.search(question, k=S.top_k_for(p))
        ctx = build_context(question, hits, qtype)
        try:
            raw = llm.chat(build_messages(question, hits, qtype, p["max_sent"], ctx=ctx), p["max_new"])
            post_type, post_sent, calcs = qtype, p["max_sent"], []
            debug = {"first": raw}
            if qtype == "calc":
                qvals = values_in_question(question)
                calcs = calc_from_values(question, qvals) if qvals else []   # 質問が数を与えていれば、それで計算する
                if calcs:
                    debug["question_values"] = qvals
                if not calcs:
                    calcs = run_calc_all(raw, p["n_calc"]) if "<calc>" in raw else []
                if not calcs:                           # <calc> は無いが、文の中に式を書いていれば、それをこちらで計算し直す
                    calcs = exprs_in_text(raw, p["n_calc"])
                if not calcs:                           # 式も無い → 数値だけ書き出させて、計算はこちらでする
                    vals_raw = llm.chat(build_extract_messages(question, ctx), 128)
                    debug["extract"] = vals_raw
                    calcs = calc_from_values(question, clean_values(parse_values(vals_raw))) or exprs_in_text(vals_raw, p["n_calc"])
                    raw = raw + " ||数値|| " + vals_raw
                multi = p["parts"] >= 2
                if calcs:
                    post_type = "explain" if multi else "fact"
                    post_sent = max(2, p["parts"]) if multi else 3
                    raw = llm.chat(build_calc_answer_messages(question, hits, calcs, multi, post_sent),
                                   64 + 40 * p["parts"] if multi else S.MAX_NEW_TOKENS["fact"])
                else:                                   # 計算できなければ事実として答え直す（計算の結果が2文目に来ることがあるので2文まで残す）
                    post_type, post_sent = "explain", 2
                    raw = llm.chat(build_messages(question, hits, "fact"), S.MAX_NEW_TOKENS["fact"] + 32)
            ans = postprocess(raw, post_type, post_sent)
            cur = raw                                   # 直しのときにモデルへ見せる「今の答え」
            if calcs:
                ans = ensure_calc_values(ans, calcs)
                debug["calcs"] = [f"{e} = {v}" for e, v in calcs]
            elif S.REPAIR and S.remaining() > 60 and is_refusal(ans):   # 「記載されていません」→ 手作りデータも足して探し直す
                q2 = relax_query(question)              # 「〜に掲載された」「2027年度」などを外して探し直す
                seen, hits2 = set(), []
                for h in self.retriever.search_manual(q2) + self.retriever.search(q2, k=S.TOP_K) + hits:
                    if h[0].id not in seen and len(hits2) < S.TOP_K:
                        seen.add(h[0].id)
                        hits2.append(h)
                debug["relaxed"] = q2
                if seen - {c.id for c, _ in hits}:
                    ctx2 = build_context(question, hits2, post_type)
                    msgs2 = build_messages(question, hits2, post_type, post_sent, ctx=ctx2)
                    msgs2[-1]["content"] += "\n（答えは資料の中にあります。「記載されていません」とは書かず、資料から答えてください。）"
                    raw2 = llm.chat(msgs2, p["max_new"] + 64)
                    ans2 = drop_refusal_sentences(postprocess(raw2, "explain", 3))
                    debug["refusal_retry"] = raw2[:300]
                    if ans2 and not is_refusal(ans2):
                        ans, raw, ctx, hits, cur = ans2, raw + " ||探し直し|| " + raw2, ctx2, hits2, raw2
            if not calcs and S.REPAIR and S.remaining() > 60:   # 答えの形を点検し、だめなときだけ1回直させる
                words = shape_missing(question, ans)
                if words:
                    msgs = build_messages(question, hits, post_type, post_sent, ctx=ctx)
                    msgs += [{"role": "assistant", "content": cur}, {"role": "user", "content": repair_message(question, words)}]
                    raw2 = llm.chat(msgs, p["max_new"])
                    ans2 = postprocess(raw2, "fact", 3)
                    if ans2 and not shape_missing(question, ans2):
                        ans, raw = ans2, raw + " ||直し|| " + raw2
        except Exception as e:
            self.log(f"生成に失敗: {type(e).__name__}: {str(e)[:120]}")
            raw, ans, debug = "", "", {}
        if not ans.strip():
            ans = extractive(ctx, question)
        return {"question": question, "answer": ans[:S.MAX_ANSWER_CHARS], "raw": raw, "qtype": qtype,
                "parts": p["parts"], "sources": [c.source for c, _ in hits], "context": ctx, "debug": debug}