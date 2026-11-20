"""① 振り分け：質問を fact（事実）/ calc（計算）/ explain（説明）に分類する。

単語を1つずつ足していく方式は、新しい言い回しに弱い。そこで「質問の形」で判定する。

  calc    : 計算の言葉（または「〜より高い／多い」のような比べる言い方）と 数を答えさせる言葉 の両方がある
            （「組み合わせる」「役割と合わせて」のような計算でない「合わせ」を除く）
  explain : (a) 説明を求める言葉がある（どのような・とは・理由 など）
            (b) 依頼の形（〜してください。「答えてください」は除く）で、短い答えを求める疑問詞も列挙の依頼も無い
                → 「述べて」「整理して」「比較して」なども、単語を足さずに説明になる
            （「挙げて」「順番に」のような列挙は名前を並べるだけなので fact のまま）
  fact    : それ以外（何号館・いつ・いくつ など、短く答えられる質問）

さらに plan() で、答えの部分が複数ある質問（それぞれ／「Aと、B」）は説明として答えさせ、
部分の数に合わせて出力の長さを広げる。
"""
import re

CALC_WORDS = ("合計", "合わせて", "合わせる", "合わせた", "合算", "差", "何倍", "割合", "平均", "総額",
              "何日後", "何日前", "何年後", "何時間", "計算", "足すと", "足した", "引くと", "引いた",
              "4年間で", "四年間で")
NUM_CUES = ("いくら", "いくつ", "何人", "何名", "何円", "何％", "何%", "何倍", "何日", "何年", "何時間",
            "人数", "金額", "総額", "費用", "合計", "合算", "差", "割合", "平均")
EXPLAIN_WORDS = ("どのような", "どんな", "特徴", "なぜ", "理由", "説明", "とは", "教えて",
                 "について", "違い", "どういう")

# 比べる言い方：「AはBより〜高い／多い」は引き算。数を答えさせる言葉と一緒なら計算として扱う
_COMPARE = re.compile(r"より.{0,20}?(高い|安い|多い|少ない|大きい|小さい|長い|短い|高く|安く|多く|少なく)")

# 依頼の形：「〜てください」「〜でください」「〜て下さい」「〜せよ」など。
# ただし「答えてください」は答えを求めているだけで、答えの形は決めないので除く
# （「名前を答えて」は短い答え、「〜と、〜を答えて」は複数部分として plan() が扱う）
_REQUEST = re.compile(r"(?<!答え)[てで](ください|下さい)|(?<!答え)せよ|しなさい")
# 短い答えで済む疑問詞（これがあれば依頼の形でも事実として扱う）
_SHORT_ASK = re.compile(r"何(号館|階|年|月|日|時|分|人|名|円|％|%|学部|学科|つ|個|件|回|歳)|いつ|いくつ|いくら|誰|どこ|どの学部|どれ")
# 列挙の依頼（名前を並べる答え）。文章で説明する答えではないので事実として扱う
_LIST_REQ = re.compile(r"挙げ|列挙|順番に|順に")

_ASK = re.compile(r"何|どこ|どの|どんな|どう|いつ|いくつ|いくら|誰")
# 質問に書かれた個数（「4段階」「2つ」「6学部」「3か国」）。答えに並べる項目の数の目安
_COUNT = re.compile(r"(?<![0-9０-９])([2-9２-９]|二|三|四|五|六|七|八|九)"
                    r"(つ|段階|学部|学科|種類|か国|ヶ国|カ国|大学|項目|か所|ヶ所|点|人|名)")
_KANJI = {"二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def listed_count(question: str) -> int:
    """質問に書かれた個数の最大値（無ければ 0）。"""
    nums = []
    for d, _ in _COUNT.findall(question):
        nums.append(_KANJI.get(d) or int(d.translate(str.maketrans("２３４５６７８９", "23456789"))))
    return max(nums, default=0)
_MULTI = ("それぞれ", "および", "ならびに")
# 「Aと、B」の並べ方。ただし「足すと、」「合わせると、」のような条件の「と」（動詞の後）は除く
_AND_COMMA = re.compile(r"(?<![すくるうつぬむぶぐ])と、")


def classify(question: str) -> str:
    has_num = any(c in question for c in NUM_CUES)
    if has_num and (any(w in question for w in CALC_WORDS) or _COMPARE.search(question)):
        return "calc"
    if any(w in question for w in EXPLAIN_WORDS):
        return "explain"
    if _REQUEST.search(question) and not _SHORT_ASK.search(question) and not _LIST_REQ.search(question):
        return "explain"
    return "fact"


# ---- 複数のことを聞く質問 ----------------------------------------------
def is_multi(question: str) -> bool:
    return any(m in question for m in _MULTI) or bool(_AND_COMMA.search(question))


def count_parts(question: str) -> int:
    """答えの部分の数の目安（1以上）。"""
    n = len(_ASK.findall(question))
    if is_multi(question):
        n = max(n, 2)
        if "それぞれ" in question and "・" in question:   # 「A・B・Cをそれぞれ」
            n = max(n, question.count("・") + 1)
    return max(n, listed_count(question), 1)


def plan(question: str) -> dict:
    """種類・部分の数・出力の上限をまとめて決める。pipeline と評価で同じものを使う。"""
    from config import CFG
    qtype = classify(question)
    parts = count_parts(question)
    if qtype == "fact" and is_multi(question):     # 複数の答えが要る事実質問は説明として答える
        qtype = "explain"
    base = CFG.max_new_tokens[qtype]
    if qtype == "explain":
        max_new = base + 80 * max(parts - 2, 0)
    else:                                          # 事実型でも「2つ」「6学部」のように並べる数が書かれていれば少し長く
        max_new = base + 16 * max(listed_count(question) - 1, 0)
    return {"qtype": qtype, "parts": parts, "max_new": max_new,
            "max_sent": max(3, parts + 1), "n_calc": min(parts, 3) if qtype == "calc" else 1}