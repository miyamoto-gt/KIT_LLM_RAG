"""① 振り分け：質問を fact（事実）/ calc（計算）/ explain（説明）に分類する。

まずはキーワードのルール。後で TF-IDF＋ロジスティック回帰の分類器に差し替える想定で、
classify(question) -> str の形だけ守る。
"""
CALC_WORDS = ("合計", "合わせて", "合わせる", "合わせた", "差", "何倍", "割合", "平均", "総額", "何日後", "何日前",
              "何年後", "何時間", "計算", "足すと", "足した", "引くと", "引いた", "4年間で", "四年間で")
EXPLAIN_WORDS = ("どのような", "どんな", "特徴", "なぜ", "理由", "説明", "とは", "教えて",
                 "について", "違い", "どういう")


def classify(question: str) -> str:
    if any(w in question for w in CALC_WORDS):
        return "calc"
    if any(w in question for w in EXPLAIN_WORDS):
        return "explain"
    return "fact"