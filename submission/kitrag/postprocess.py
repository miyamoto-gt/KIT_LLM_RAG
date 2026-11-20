"""生成文の後処理（暴走の切り落とし・前置きの除去・一覧の連結・文数の制限・計算値の補完）。"""
import re

_EN_RUN = re.compile(r"(?:[A-Za-z][A-Za-z0-9'\-]*[\s,.:;()\"/&]+){8,}")


def cut_english(text: str) -> str:
    m = _EN_RUN.search(text)
    if not m:
        return text
    head = text[:m.start()]
    if "。" in head:
        return head[:head.rindex("。") + 1]
    if "、" in head:
        return head[:head.rindex("、")] + "。"
    head = head.rstrip(" （(　")
    return head + "。" if head else text



STOP_MARKERS = ("\n質問", "\n###", "\n資料", "\n例）")
LIST_ITEM = re.compile(r"\n\s*(?:[-・*●◆]|\d+\s*[.．)）、])\s*")
# 「以下の通りです。」のような前置きだけの行（中身は次の行から続く）
PREAMBLE = re.compile(r"(以下の(通り|とおり|よう)|次の(通り|とおり|よう)|以下になります|以下です)")
LABEL_ITEM = re.compile(r"^[^。]{1,40}?[：:]\s*\S")     # 「1966年：経営工学科」のような見出し付きの行


# 「資料[2]の」「資料[1]と[3]から、」のような出典への言及（答えの中身ではない）
_REF = re.compile(r"(与えられた)?資料\s*[\[［]\d+[\]］](\s*[とや、・]\s*(資料\s*)?[\[［]?\d+[\]］]?)*"
                  r"\s*(の両方|のいずれ)?\s*(の中で|の中に|によると|によれば|では|から|に|で|の)?\s*[、,]?")
# 答えの中身ではない文（出典の話・「資料からは分からない」・「詳しくは〜」の案内）
_META_SENT = re.compile(r"与えられた資料|記載(があ|されて)|(資料|情報)(に|では|から)は?.{0,12}(記載|書かれ|記述)|"
                        r"同じ情報が|より詳(細|しい)な情報|詳しくは|確認(して|することを)")


def drop_meta(text: str) -> str:
    """出典への言及を外し、中身の無い文（資料の話・案内）を落とす。中身のある文が1つも無ければ元のまま。"""
    sents = [_REF.sub("", x).strip() for x in re.findall(r"[^。\n]+(?:。|$)", text)]
    sents = [x for x in sents if x.strip("。、 ")]
    keep = [x for x in sents if not _META_SENT.search(x)]
    return "".join(keep or sents).strip()


def drop_cut_tail(text: str) -> str:
    """出力の上限で途中で切れた最後の文を落とす（その前に完結した文があるときだけ）。"""
    t = text.rstrip()
    if t and not t.endswith(("。", "）", ")", "」")) and "。" in t:
        return t[:t.rindex("。") + 1]
    return t


def drop_preamble(text: str) -> str:
    """1行目が前置き（以下の通りです 等）で、その後に中身があれば、前置きの行を外す。
    事実型は1行目だけを答えにするので、前置きが残ると中身が全部消えてしまう。"""
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) >= 2 and len(lines[0]) <= 80 and PREAMBLE.search(lines[0]):
        return "\n".join(lines[1:])
    return text


def is_item(line: str) -> bool:
    return bool(LIST_ITEM.match("\n" + line)) or (bool(LABEL_ITEM.match(line)) and "。" not in line[:-1])


def strip_item(line: str) -> str:
    m = LIST_ITEM.match("\n" + line)
    return line[m.end() - 1:].strip() if m else line.strip()


def ensure_calc_values(answer: str, calcs: list) -> str:
    """計算ツールで出した値が答えに書かれていなければ、末尾に（式 = 値）を添える。
    モデルが値を書き写し忘れても、計算結果は必ず答えに残る。"""
    def digits(s):
        return re.sub(r"[,，\s円人名]", "", str(s))
    missing = [(e, v) for e, v in calcs if digits(v) not in digits(answer)]
    if not missing:
        return answer
    add = "、".join(f"{e} = {v}" for e, v in missing)
    return f"{answer}（計算：{add}）" if answer else f"{add}"


def postprocess(text: str, qtype: str, max_sent: int = 3) -> str:
    text = text.strip()                          # 先頭の改行を先に取り除く
    for m in STOP_MARKERS:                       # 次の質問を書き始める等の暴走を切る
        if m in text:
            text = text.split(m)[0]
    text = cut_english(text.strip())             # 途中から英語に切り替わったら、その手前の「。」で切る
    after = drop_preamble(text)                  # 「以下の通りです。」だけの1行目を外す
    listed = after != text                       # 前置きの後ろは一覧（記号が無くても項目とみなす）
    text = after
    if qtype in ("fact", "calc"):
        lines = [l.strip() for l in text.split("\n\n")[0].splitlines() if l.strip()]
        if len(lines) >= 2 and listed and all(len(l) <= 60 for l in lines):
            text = "、".join(strip_item(l).rstrip("。") for l in lines) + "。"
        elif len(lines) >= 2 and is_item(lines[0]):   # 一覧で答えている → 続く項目を「、」でつなぐ
            items = []
            for l in lines:
                if not is_item(l):
                    break
                items.append(strip_item(l).rstrip("。"))
            text = "、".join(items) + "。"
        else:
            text = lines[0] if lines else ""
        text = re.sub(r"^(回答|答え)[:：]\s*", "", text)
        text = drop_meta(drop_cut_tail(text))
        first = re.findall(r"[^。]+。", text)
        if first and qtype == "fact":            # 事実は最初の1文（短すぎるときは2文）
            text = "".join(first[:2]) if len(first[0]) < 12 else first[0]
    if qtype == "explain":
        text = text.split("\n\n資料")[0]
        parts, items = [], []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if is_item(line):                    # 箇条書き・「見出し：中身」の1行
                items.append(strip_item(line).rstrip("。"))
                continue
            if items:                            # 箇条書きが終わったら「、」でつないで1文にする
                parts.append("、".join(items) + "。")
                items = []
            parts.append(line)
        if items:
            parts.append("、".join(items) + "。")
        text = drop_meta(drop_cut_tail("".join(parts).replace("。。", "。")))
        if text and not text.endswith("。"):
            text += "。"
        sentences = re.findall(r"[^。]+。", text)
        if sentences:                            # 説明は最初の max_sent 文まで（部分が多い質問ほど長く）
            text = "".join(sentences[:max_sent])
    return text.strip()
