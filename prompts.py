"""⑦ プロンプト組み立て。

sarashina2.2-3b は指示調整前のベースモデルなので、指示文よりも「お手本の続き」を書かせる形にする。
  [前置き] → [お手本（資料・質問・答え）×数件] → [本番の資料・質問] → 「答え:」
こうすると、続きとして答えを1行（説明なら数文）書く。

fact 型は、資料から質問のキーワードを含む行だけを抜き出して渡す（文脈圧縮）。
3Bモデルは正解が資料にあっても、文脈が長いと読み取れないため（q006で確認）。
RAG_COMPRESS=0 で従来方式に戻せる。
"""
import os
import re

COMPRESS = os.environ.get("RAG_COMPRESS", "1") == "1"

# 圧縮は「いつ／何年」を聞く質問だけに使う。
# 全fact型にかけると、授業料・入学金など表の資料で必要な行が落ちて悪化した（A08, A12）。
_DATE_Q = re.compile(r"いつ|何年|何月|何日|年は")


def _is_date_question(question: str) -> bool:
    return bool(_DATE_Q.search(question))

HEADER = "以下は、金沢工業大学に関する資料を読んで質問に答えた記録です。答えは資料に書かれている内容だけを使います。\n\n"

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

"""

CALC_EXAMPLES = """資料:
[1] 【学費の例】 前期授業料 750,000円、後期授業料 750,000円
質問: 前期と後期の授業料の合計はいくらですか？
計算式: <calc>750000 + 750000</calc>

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
    if question and COMPRESS:
        kws = _keywords(question)
        comp = [c for c in compress_all([chunk.text for chunk, _ in ordered], kws) if c]
        if comp:                                    # 全部空なら従来方式にフォールバック
            return "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(comp))
    return "\n".join(f"[{i + 1}] {_one_line(chunk.text)}" for i, (chunk, _) in enumerate(ordered))


def build_prompt(question: str, hits: list, qtype: str) -> str:
    examples = {"fact": FACT_EXAMPLES, "explain": EXPLAIN_EXAMPLES, "calc": CALC_EXAMPLES}[qtype]
    tail = "計算式:" if qtype == "calc" else "答え:"
    use = qtype == "fact" and _is_date_question(question)
    ctx = format_context(hits, question if use else None)
    return HEADER + examples + f"資料:\n{ctx}\n質問: {question}\n{tail}"


def build_calc_answer_prompt(question: str, hits: list, expr: str, value) -> str:
    return (HEADER + FACT_EXAMPLES
            + f"資料:\n{format_context(hits)}\n[計算] {expr} = {value}\n質問: {question}\n答え:")