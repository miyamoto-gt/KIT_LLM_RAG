"""⑦ プロンプト組み立て。

sarashina2.2-3b は指示調整前のベースモデルなので、指示文よりも「お手本の続き」を書かせる形にする。
  [前置き] → [お手本（資料・質問・答え）×数件] → [本番の資料・質問] → 「答え:」
こうすると、続きとして答えを1行（説明なら数文）書く。
"""

HEADER = "以下は、金沢工業大学に関する資料を読んで質問に答えた記録です。答えは資料に書かれている内容だけを使います。\n\n"

FACT_EXAMPLES = """資料:
[1] 【扇が丘キャンパス／27号館】 施設と階： ・学生ラウンジ（1F） ・コンビニエンスストア（2F）
質問: コンビニエンスストアは何号館の何階にありますか？
答え: 27号館の2階です。

資料:
[1] 【扇が丘キャンパス／シャトルバス発着所】 やつかほリサーチキャンパス行きのシャトルバスは、23号館と24号館の間から発着する。
質問: やつかほリサーチキャンパス行きのシャトルバスはどこから出ますか？
答え: 23号館と24号館の間です。

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


def format_context(hits: list) -> str:
    # 類似度の高い資料ほど質問の直前に来るよう、逆順に並べる（小さいモデルは近い情報を使いやすい）
    ordered = list(reversed(hits))
    return "\n".join(f"[{i + 1}] {_one_line(chunk.text)}" for i, (chunk, _) in enumerate(ordered))


def build_prompt(question: str, hits: list, qtype: str) -> str:
    examples = {"fact": FACT_EXAMPLES, "explain": EXPLAIN_EXAMPLES, "calc": CALC_EXAMPLES}[qtype]
    tail = "計算式:" if qtype == "calc" else "答え:"
    return HEADER + examples + f"資料:\n{format_context(hits)}\n質問: {question}\n{tail}"


def build_calc_answer_prompt(question: str, hits: list, expr: str, value) -> str:
    return (HEADER + FACT_EXAMPLES
            + f"資料:\n{format_context(hits)}\n[計算] {expr} = {value}\n質問: {question}\n答え:")