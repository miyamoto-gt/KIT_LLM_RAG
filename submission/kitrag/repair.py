"""答えを規則で点検して、だめなところだけモデルに直させる部品（小さいモデルの弱点をしくみで補う）。

1) 質問を分ける（decompose）
   「AとBは、それぞれ〜ですか」のように、同じことを複数の対象について聞く質問は、
   「Aは〜ですか」「Bは〜ですか」に分けて、1つずつ検索・生成してからつなぐ。
   小さいモデルは、まとめて聞くと片方しか答えなかったり、片方の資料に引っ張られたりする。

2) 単位の点検（unit_missing）
   「何号館」「何校地」「何大学」「何人」「いくつ」のように、答えの形が決まっている質問で、
   答えにその形（「〇号館」「校地」「数＋大学」など）が無ければ、足りないものを伝えて1回だけ答え直させる。
   点検に通った答えは直さないので、API の呼び出しが増えるのは、だめだった問題だけ。
"""
import re

# ---- 1) 質問を分ける ----------------------------------------------------------
# 「A と B は、それぞれ …」「A・B・C は それぞれ …」（主題として対象を並べ、それぞれ同じことを聞く形だけ）
_EACH = re.compile(r"^(?P<items>.+?)(?P<p>は|では|について)、?\s*それぞれ(?P<rest>[^の].*)$")
_SPLIT = re.compile(r"と、?|・|、|および|ならびに")


def decompose(question: str) -> list:
    """分けられる質問なら [部分の質問, …]（2〜3個）、分けられなければ []。"""
    m = _EACH.match(question.strip())
    if not m:
        return []
    items = [x.strip() for x in _SPLIT.split(m.group("items")) if x.strip()]
    if not 2 <= len(items) <= 3 or any(len(x) < 2 or re.search(r"\d+\s*(つ|個|校|大学|学部|学科|キャンパス)$", x)
                                        for x in items):
        return []
    rest = m.group("rest").lstrip("、 ")
    return [f"{x}{m.group('p')}{rest}" for x in items]


# ---- 2) 単位の点検 ------------------------------------------------------------
_NUM = r"[0-9０-９一二三四五六七八九十百千]"
_UNIT_RULES = [   # (質問の形, 答えに必要な形, 伝える言葉)
    (r"何号館", rf"{_NUM}+\s*号館", "「〇号館」という号館の番号"),
    (r"何階", rf"{_NUM}+\s*(階|F)", "「〇階」という階"),
    (r"何校地", r"校地", "「〇校地」という校地の名前"),
    (r"何(大学|校)", rf"{_NUM}+\s*(大学|校)", "「〇大学」「〇校」という数"),
    (r"何(人|名)", rf"{_NUM}[0-9,，]*\s*(人|名)", "「〇人」「〇名」という人数"),
    (r"何学部", rf"{_NUM}+\s*(学部|つ)", "「〇学部」という数"),
    (r"何円|いくら", rf"{_NUM}[0-9,，万千]*\s*円", "「〇円」という金額"),
    (r"いくつ|何つ|何個|何件|何回", rf"{_NUM}+\s*(つ|個|件|回|校|大学|学部|学科|か所|カ所)", "数"),
    (r"何年(?!間|度生|次)|いつ", rf"{_NUM}+\s*年|昭和|平成|令和", "年"),
]


def unit_missing(question: str, answer: str):
    """答えに、質問が求める形が無ければ、その説明（無ければ None）。"""
    for q_pat, a_pat, words in _UNIT_RULES:
        if re.search(q_pat, question):
            return None if re.search(a_pat, answer or "") else words
    return None


def repair_message(question: str, words: str) -> str:
    return (f"今の答えには{words}が書かれていません。資料から探して、質問「{question}」に、"
            f"{words}を含めて1文で答え直してください。")