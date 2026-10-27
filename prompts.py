"""⑦ プロンプト組み立て。質問タイプごとに出力の指示を変える。"""

INSTRUCTION = {
    "fact": "資料に基づき、答えだけを短く書いてください。前置きや説明は不要です。",
    "explain": "資料に基づき、2〜3文で簡潔に説明してください。",
    "calc": "資料から必要な数値を探し、計算式だけを <calc>式</calc> の形で書いてください。",
}

CALC_EXAMPLE = (
    "例）質問：前期と後期の授業料の合計は？\n"
    "資料：前期授業料 750,000円、後期授業料 750,000円\n"
    "計算式：<calc>750000 + 750000</calc>\n\n"
)


def format_context(hits: list) -> str:
    return "\n\n".join(f"### 資料{i + 1}\n{chunk.text}" for i, (chunk, _) in enumerate(hits))


def build_prompt(question: str, hits: list, qtype: str) -> str:
    head = "以下は金沢工業大学に関する資料です。\n\n" + format_context(hits) + "\n\n"
    if qtype == "calc":
        return head + CALC_EXAMPLE + f"{INSTRUCTION['calc']}\n質問：{question}\n計算式："
    return head + f"{INSTRUCTION[qtype]}\n質問：{question}\n回答："


def build_calc_answer_prompt(question: str, hits: list, expr: str, value) -> str:
    return ("以下は金沢工業大学に関する資料です。\n\n" + format_context(hits) + "\n\n"
            f"質問：{question}\n計算式：{expr} = {value}\n"
            "計算結果を使い、単位を付けて答えだけを短く書いてください。\n回答：")
