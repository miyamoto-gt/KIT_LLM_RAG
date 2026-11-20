"""プロンプト（チャット形式）と、渡す資料の組み立て。

API の sarashina2.2-3b は instruct 版なので、system の指示 ＋ お手本の会話（user/assistant）＋ 本番の質問 で頼む。
お手本は手作りデータ（キャンパスマップ・学費）から作ったもので、公開問題の文面・答えは使っていない。
資料は narrow_context() で、質問に関係する文（とその前後）だけに絞って渡す。
"""
import math
import re
from collections import defaultdict

from . import settings as S
from .router import count_parts

_DATE_Q = re.compile(r"いつ|何年|何月|何日|年は")


def _is_date_question(question: str) -> bool:
    return bool(_DATE_Q.search(question))


FACT_EXAMPLES = """資料:
[1] 【扇が丘キャンパス／27号館】 施設と階： ・学生ラウンジ（1F） ・コンビニエンスストア（2F）
質問: コンビニエンスストアは何号館の何階にありますか？
答え: 27号館の2階です。

資料:
[1] 【扇が丘キャンパス／シャトルバス発着所】 やつかほリサーチキャンパス行きのシャトルバスは、23号館と24号館の間から発着する。
質問: やつかほリサーチキャンパス行きのシャトルバスはどこから出ますか？
答え: 23号館と24号館の間です。

資料:
[1] 【扇が丘キャンパス／47号館】 施設： ・KIT国際交流会館
質問: KIT国際交流会館は何号館にありますか？
答え: 47号館です。

"""

EXPLAIN_EXAMPLES = """資料:
[1] 【扇が丘キャンパス／23号館】 施設と階： ・CDIOイノベーション＆デザインスタジオ（1F） ・学生ステーション（1F） ・基礎英語教育センター（2F・3F） ・留学支援・国際交流センター（2F） ・数理工教育研究センター（4F・5F）
質問: 23号館にはどのような施設がありますか？
答え: 23号館には、1階に学生ステーションとCDIOイノベーション＆デザインスタジオがあります。2階には留学支援・国際交流センター、2〜3階に基礎英語教育センター、4〜5階に数理工教育研究センターがあります。

資料:
[1] 【扇が丘キャンパス／27号館】 施設と階： ・学生ラウンジ（1F） ・コンビニエンスストア（2F）
[2] 【扇が丘キャンパス／シャトルバス発着所】 やつかほリサーチキャンパス行きのシャトルバスは、23号館と24号館の間から発着する。
質問: 学生ラウンジとシャトルバス発着所はそれぞれどこにありますか？
答え: 学生ラウンジは27号館の1階です。シャトルバス発着所は23号館と24号館の間で、やつかほリサーチキャンパス行きのバスが発着します。

"""

CALC_EXAMPLES = """資料:
[1] 【学費の例】 前期授業料 750,000円、後期授業料 750,000円
質問: 前期と後期の授業料の合計はいくらですか？
計算式: <calc>750000 + 750000</calc>

資料:
[1] 【学費の例】 前期授業料 750,000円、後期授業料 750,000円、入学金 200,000円
質問: 1年間の授業料の合計と、入学金を含めた合計はそれぞれいくらですか？
計算式: <calc>750000 + 750000</calc> <calc>750000 + 750000 + 200000</calc>

"""


def _one_line(text: str) -> str:
    """チャンク内の改行を空白にまとめ、資料1件を1段落にする（続きを書かれにくくする）。"""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


# ---- 文脈圧縮 ------------------------------------------------------------

def _split_lines(text: str) -> list:
    """年表の日付・文末で区切って、行（セグメント）のリストにする。"""
    t = _one_line(text)
    t = re.sub(r" - (\d{4}年)", r"\n- \1", t)                       # 西暦の年表
    t = re.sub(r" ?(昭和|平成|令和) ?(\d+) ?年 ?(\d+) ?月", r"\n\1\2年\3月", t)   # 和暦の年表（月まであるものだけ）
    t = re.sub(r"。", "。\n", t)                                     # 文の区切り
    return [s.strip() for s in t.split("\n") if s.strip()]


_DATE_HEAD = re.compile(
    r"^-?\s*((?:\d{4}年|(?:昭和|平成|令和)\d+年)\s*(?:\d+\s*月)?\s*(?:\d+\s*日)?)\s*(.+)$")


def _date_last(seg: str) -> str:
    """「1966年4月1日 経営工学科設置」→「経営工学科設置（1966年4月1日）」。

    3Bモデルはキーワードの「直後」にある日付を答えに使いがちなので、
    年表の行は日付を項目の後ろに回して、直後の日付＝その項目の日付にする（q006で確認）。
    """
    m = _DATE_HEAD.match(seg)
    if not m:
        return seg
    date, body = m.group(1).replace(" ", ""), m.group(2).strip()
    return f"{body}（{date}）"


def _keywords(question: str) -> list:
    return re.findall(r"[一-龥ァ-ヶーA-Za-z0-9]{2,}", question)


def _score(seg: str, kws: list) -> int:
    return sum(len(k) for k in kws if k in seg)


def compress_all(texts: list, kws: list, ratio: float = 0.8, win: int = 1) -> list:
    """全資料を通して、キーワード一致スコアが最高点の ratio 倍以上の行だけを残す。

    資料ごとに上位N行を取ると「工学部」「設置」だけ一致する弱い行まで混ざり、
    年号の候補が増えて誤答した（q006で確認）。そのため閾値は全資料共通の最高点で決める。
    残る行がない資料は空文字を返す（＝プロンプトから外す）。
    """
    segs_list = [_split_lines(t) for t in texts]
    best = max((_score(s, kws) for segs in segs_list for s in segs), default=0)
    if best == 0:
        return ["" for _ in texts]
    out = []
    for segs in segs_list:
        keep = set()
        for i, s in enumerate(segs):
            if _score(s, kws) >= best * ratio:
                keep.update(range(max(0, i - win), min(len(segs), i + win + 1)))
        out.append(" ／ ".join(_date_last(segs[i]) for i in sorted(keep)))   # お手本と同じく1資料1行
    return out


# ---- 組み立て ------------------------------------------------------------

def format_context(hits: list, question: str = None) -> str:
    # 類似度の高い資料ほど質問の直前に来るよう、逆順に並べる（小さいモデルは近い情報を使いやすい）
    ordered = list(reversed(hits))
    if question:
        kws = _keywords(question)
        comp = [c for c in compress_all([chunk.text for chunk, _ in ordered], kws) if c]
        if comp:                                    # 全部空なら従来方式にフォールバック
            return "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(comp))
    return "\n".join(f"[{i + 1}] {_one_line(chunk.text)}" for i, (chunk, _) in enumerate(ordered))


# ---- 資料を質問に関係する文だけに絞る ---------------------------------------

def _bigrams(text: str) -> set:
    t = "".join(text.split())
    return {t[i:i + 2] for i in range(len(t) - 1)}


_KANA2 = re.compile(r"^[ぁ-んー、。？！・「」（）\s]{2}$")   # 「です」「には」のような、どこにでも出る2文字は使わない


def narrow_context(hits: list, question: str, keep: int = None, win: int = None) -> str:
    """検索で選んだ資料を文に分け、質問との関連度が高い文（と前後 win 文）だけを残す。

    関連度：質問の文字bigramが文に含まれるかを、珍しい bigram ほど重く数える（資料の中での IDF）。
    ・ページのタイトル（【…】の行）は残す。手作りデータの短いチャンクは丸ごと残す
    ・どの文も質問と重ならなければ ""（呼び出し側で全文を渡す）
    """
    keep = S.NARROW_KEEP if keep is None else keep
    win = S.NARROW_WIN if win is None else win
    q = {b for b in _bigrams(question) if not _KANA2.match(b)}
    ordered = list(reversed(hits))                      # 関連の高い資料ほど質問の直前に来る並び（従来どおり）
    docs = []
    for chunk, _ in ordered:
        text = chunk.text
        head = text.split("\n", 1)[0] if text.startswith("【") else ""
        body = text.split("\n", 1)[1] if head and "\n" in text else text
        docs.append((_one_line(head), _split_lines(body), chunk))
    segs = [s for _, ss, _ in docs for s in ss]
    if not q or not segs:
        return ""
    n = len(segs)
    df = {b: sum(1 for s in segs if b in s) for b in q}
    weight = {b: math.log(1 + n / df[b]) for b in q if df[b]}
    scored = [(sum(w for b, w in weight.items() if b in s), di, si)
              for di, (_, ss, _) in enumerate(docs) for si, s in enumerate(ss)]
    if not scored or max(scored)[0] <= 0:
        return ""
    kept = defaultdict(set)
    for sc, di, si in sorted(scored, reverse=True)[:keep]:
        if sc <= 0:
            continue
        for j in range(si - win, si + win + 1):
            if 0 <= j < len(docs[di][1]):
                kept[di].add(j)
    lines = []
    for di, (head, ss, chunk) in enumerate(docs):
        if di not in kept:
            continue
        if chunk.type == "manual" and len(chunk.text) <= 300:
            lines.append(_one_line(chunk.text))         # 手作りデータは短く完結しているので丸ごと
            continue
        body = " ／ ".join(_date_last(ss[j]) for j in sorted(kept[di]))
        lines.append(f"{head} {body}".strip())
    return "\n".join(f"[{i + 1}] {l}" for i, l in enumerate(lines))


def build_context(question: str, hits: list, qtype: str) -> str:
    """プロンプトに入れる資料の文字列。評価（rag_check）もこれを使い、本番と同じ資料を測る。"""
    if qtype == "fact" and _is_date_question(question):
        return format_context(hits, question)           # 日付の質問は従来の圧縮（実績あり）
    if S.NARROW:
        parts = count_parts(question)
        ctx = narrow_context(hits, question, keep=S.NARROW_KEEP + 2 * max(parts - 1, 0))
        if ctx:
            return ctx
    return format_context(hits)


# ---- チャット形式のメッセージ --------------------------------------------------

SYSTEM = ("あなたは金沢工業大学（KIT）に関する質問に答えるアシスタントです。"
          "与えられた資料だけを根拠に、日本語で答えてください。資料に書かれていないことは付け足さないでください。"
          "資料に答えがそのまま無い場合は、資料から分かる範囲で最も関係する内容を簡潔に答えてください。"
          "「資料[1]」のような資料の番号や「資料によると」は書かず、答えの内容だけを書いてください。")

INSTRUCTION = {
    "fact": "質問への答えだけを、1文で簡潔に書いてください。前置き（「以下の通りです」など）や説明は書かないでください。",
    "explain": ("質問の各部分に順番に答えてください。各部分は1文ずつ、資料の言葉を使って簡潔に書き、"
                "全体で{max_sent}文以内にしてください。前置きは書かないでください。"),
    "calc": ("質問に答えるのに必要な数値を資料から探し、計算式だけを <calc>式</calc> の形で書いてください。"
             "計算が2つ必要なら <calc> を2つ書いてください。答えの文章は書かないでください。"),
}


def _examples(text: str) -> list:
    """お手本の文字列を (user, assistant) の組に分ける。"""
    pairs = []
    for block in [b.strip() for b in text.strip().split("\n\n") if b.strip()]:
        m = re.search(r"\n(答え|計算式):\s*", block)
        if m:
            pairs.append((block[:m.start()].strip(), block[m.end():].strip()))
    return pairs


_EXAMPLES = {"fact": _examples(FACT_EXAMPLES), "explain": _examples(EXPLAIN_EXAMPLES),
             "calc": _examples(CALC_EXAMPLES)}


def build_messages(question: str, hits: list, qtype: str, max_sent: int = 3, ctx: str = None) -> list:
    ctx = build_context(question, hits, qtype) if ctx is None else ctx
    msgs = [{"role": "system", "content": SYSTEM + INSTRUCTION[qtype].format(max_sent=max_sent)}]
    for u, a in _EXAMPLES[qtype]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": f"資料:\n{ctx or '（資料なし）'}\n質問: {question}"})
    return msgs


def build_calc_answer_messages(question: str, hits: list, calcs: list, multi: bool, max_sent: int = 3) -> list:
    """計算結果を添えて答えを書かせる。答えの部分が複数なら説明型で複数文。"""
    qtype = "explain" if multi else "fact"
    calc_lines = "\n".join(f"[計算] {e} = {v}" for e, v in calcs)
    ctx = build_context(question, hits, "calc")
    msgs = build_messages(question, hits, qtype, max_sent, ctx=ctx)
    msgs[0]["content"] += "[計算] の結果は正しいので、その数値をそのまま使って答えてください。"
    msgs[-1]["content"] = f"資料:\n{ctx or '（資料なし）'}\n{calc_lines}\n質問: {question}"
    return msgs
