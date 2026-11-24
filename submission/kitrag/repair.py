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


# 「…、1982年・1993年・1995年にそれぞれ関係する…」（文の途中に「・」で並べた対象＋それぞれ）
_EACH_MID = re.compile(r"^(?P<pre>.*?)(?P<items>[^、。・]+(?:・[^、。・]+){1,2})(?P<p>は|では|について|に)、?\s*それぞれ(?P<rest>[^の].*)$")
_BAD_ITEM = re.compile(r"\d+\s*(つ|個|校|大学|学部|学科|キャンパス)$")


def decompose(question: str) -> list:
    """分けられる質問なら [部分の質問, …]（2〜3個）、分けられなければ []。"""
    q = question.strip()
    m = _EACH.match(q)
    if m:
        items = [x.strip() for x in _SPLIT.split(m.group("items")) if x.strip()]
        if 2 <= len(items) <= 3 and not any(len(x) < 2 or _BAD_ITEM.search(x) for x in items):
            rest = m.group("rest").lstrip("、 ")
            return [f"{x}{m.group('p')}{rest}" for x in items]
    m = _EACH_MID.match(q)
    if m:
        items = [x.strip() for x in m.group("items").split("・") if x.strip()]
        if 2 <= len(items) <= 3 and not any(len(x) < 2 or _BAD_ITEM.search(x) for x in items):
            rest = m.group("rest").lstrip("、 ")
            return [f"{m.group('pre')}{x}{m.group('p')}{rest}" for x in items]
    return []


# ---- 2) 単位の点検 ------------------------------------------------------------
_NUM = r"[0-9０-９一二三四五六七八九十百千]"
_UNIT_RULES = [   # (質問の形, 答えに必要な形, 伝える言葉)
    (r"何号館", rf"{_NUM}+\s*号館", "「〇号館」という号館の番号"),
    (r"どこにあり|どこで(行|開|受)|場所はどこ", r"号館|階|キャンパス|校地|棟|館|室|センター内|市|町", "場所（号館・階・キャンパスなど）"),
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


# 「6学部をすべて挙げ」「2学科を挙げて」のように数を指定して名前を並べさせる質問で、答えの名前が足りないか
_KANJI_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_LIST_Q = re.compile(r"([0-9０-９一二三四五六七八九十]+)\s*(?:つの)?(学部|学科|専攻|研究科|キャンパス|校地|センター)"
                     r"(?:の名前|名)?(?:を|は)?、?(?:すべて|全て|それぞれ)?(?:挙げ|答え|示し|書い|列挙)")


def count_missing(question: str, answer: str):
    """数を指定して名前を挙げさせる質問で、答えに出てくる名前がその数より少なければ、その説明（無ければ None）。"""
    best = None
    for d, noun in _LIST_Q.findall(question):
        n = _KANJI_NUM.get(d) or int(d.translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        if 2 <= n <= 10 and (best is None or n > best[0]):
            best = (n, noun)
    if not best:
        return None
    n, noun = best
    names = {m for m in re.findall(rf"([一-龥ァ-ヶーA-Za-z・]{{1,12}}){noun}", answer or "")
             if not re.fullmatch(r"(各|全|同|本|他|系|情報系|理工系|の)", m)}
    return None if len(names) >= n else f"{n}つの{noun}の名前をすべて（資料に書かれている名前で）"


def shape_missing(question: str, answer: str):
    """答えの形の点検（単位・数の形 → 名前の数）。足りないものの説明か None。"""
    return unit_missing(question, answer) or count_missing(question, answer)


def repair_message(question: str, words: str) -> str:
    return (f"今の答えには{words}が書かれていません。資料から探して、質問「{question}」に、"
            f"{words}を含めて1文で答え直してください。")


# ---- 3) 拒否の点検 ------------------------------------------------------------
REFUSAL = re.compile(r"記載(され|が)(てい)?(ません|ない|ありません)|書かれていません|見当たりません|わかりません|分かりません|"
                     r"特定でき(ません|ない)|含まれていません|情報はありません|読み取れません|不明です|"
                     r"(回答|お答え|判断|断定)(すること|は)?(は|が)?できません|答えられません")


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer or ""))


def drop_refusal_sentences(answer: str) -> str:
    """拒否の文（「記載されていません」など）と、その後の「確認する必要があります」などの案内を外す。何も残らなければ空。"""
    answer = re.sub(r"[（(][^）)]*(答えがあります|資料から答え|とは書かず)[^）)]*[）)]。?", "", answer or "")   # 指示の言い写し
    sents = re.findall(r"[^。]+。?", answer)
    keep = [x for x in sents if not REFUSAL.search(x) and not re.search(r"確認する必要|確定することはできません|他の資料", x)]
    return "".join(keep).strip()


# 探し直し用に、質問から「どの資料に載っているか」「年度」などの絞り込みを外す（クエリの緩和）
_RELAX = [
    r"(CAMPUS\s*GUIDE|キャンパスガイド|入試ガイド|入試制度一覧|研究センター一覧|公式ページ|奨学金ページ|海外留学ページ)[^、。？]*?(に|で|の)(掲載|記載|紹介|案内|示)[^、。？]*?(された|されている|される|された|の)?",
    r"^[^、。？]{0,25}(ページ|ガイド|GUIDE\s*\d*|一覧)(で|では)、",
    r"[^、。？]*?(によると|によれば|の表現に沿って)、?",
    r"(\d{4}|令和\d+|平成\d+)年度?(入試|入学|の)?",
]


def relax_query(question: str) -> str:
    q = question
    for pat in _RELAX:
        q = re.sub(pat, "", q)
    q = re.sub(r"^[、\s]+", "", q).strip()
    return q if len(q) >= 6 else question