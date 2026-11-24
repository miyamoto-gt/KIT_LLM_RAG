"""計算ツール：モデルが書いた <calc>式</calc> を、四則演算だけ許して安全に計算する。"""
import ast
import operator
import re

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def safe_calc(expr: str):
    expr = expr.replace(",", "").replace("，", "").replace("×", "*").replace("÷", "/")
    expr = expr.translate(str.maketrans("０１２３４５６７８９＋－＊／（）", "0123456789+-*/()"))
    value = _eval(ast.parse(expr, mode="eval"))
    return int(value) if float(value).is_integer() else round(value, 4)


def run_calc_all(text: str, limit: int = 3) -> list:
    """閉じた <calc>…</calc> をすべて計算して [(式, 結果)] を返す（最大 limit 個）。
    閉じたタグが無ければ run_calc_tags と同じく最初の1つを試す。"""
    out = []
    for expr in re.findall(r"<calc>(.*?)</calc>", text, re.S)[:limit]:
        try:
            out.append((expr.strip(), safe_calc(expr)))
        except Exception:
            pass
    if not out:
        one = run_calc_tags(text)
        out = [one] if one else []
    return out


def run_calc_tags(text: str):
    """最初の <calc>…</calc> を計算して (式, 結果) を返す。無ければ None。"""
    m = re.search(r"<calc>(.*?)(?:</calc>|$)", text, re.S | re.M)
    expr = m.group(1) if m else (text.strip().splitlines() or [""])[0]  # タグ無しなら1行目を試す
    try:
        return expr.strip(), safe_calc(expr)
    except Exception:
        return None


# ---- 数値を書き出させて、計算はこちらでする（instruct 版は <calc> を書かずに暗算して間違えることがある）----
_NUM_RE = re.compile(r"(\d[\d,，]*(?:\.\d+)?)\s*(万)?\s*(\d[\d,，]*)?\s*(千)?")
DIFF_Q = re.compile(r"差|違いは|より.{0,12}(高|低|多|少|安|大き|小さ|長|短)|引く|引いた|引くと")
RATIO_Q = re.compile(r"何倍")
AVG_Q = re.compile(r"平均")


def parse_number(text: str):
    """「1,515,000円」「171万5千円」「20万円」「6つ」などを数にする。読めなければ None。"""
    t = text.translate(str.maketrans("０１２３４５６７８９，", "0123456789,"))
    m = _NUM_RE.search(t)
    if not m:
        return None
    a = float(m.group(1).replace(",", "").replace("，", ""))
    if m.group(2):                                     # 万
        a *= 10000
        if m.group(3):
            b = float(m.group(3).replace(",", ""))
            a += b * (1000 if m.group(4) else 1)
    elif m.group(4):                                   # 千
        a *= 1000
    return int(a) if a.is_integer() else a


def parse_values(text: str, limit: int = 6) -> list:
    """「項目: 数値」の行から [(項目, 数)]。"""
    out = []
    for line in text.splitlines():
        if not re.search(r"[:：]", line):
            continue
        name, val = re.split(r"[:：]", line, 1)
        v = parse_number(val)
        if v is not None:
            out.append((name.strip(" ・-*"), v))
    return out[:limit]


def calc_from_values(question: str, values: list):
    """質問の言い方から計算の種類を決めて計算する。[(式, 結果)]（できなければ []）。"""
    nums = [v for _, v in values]
    if len(nums) < 2:
        return []
    fmt = lambda x: f"{x:,}" if isinstance(x, int) else str(x)
    if DIFF_Q.search(question):
        a, b = max(nums[:2]), min(nums[:2])
        expr, val = f"{fmt(a)} - {fmt(b)}", a - b
    elif RATIO_Q.search(question):
        a, b = nums[0], nums[1]
        if not b:
            return []
        expr, val = f"{fmt(a)} ÷ {fmt(b)}", round(a / b, 2)
    elif AVG_Q.search(question):
        expr, val = f"({' + '.join(fmt(x) for x in nums)}) ÷ {len(nums)}", round(sum(nums) / len(nums), 2)
    else:
        expr, val = " + ".join(fmt(x) for x in nums), sum(nums)
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    return [(expr, val)]


# ---- モデルが文の中に書いた式（「1,515,000円 - 1,315,000円 = …」）を見つけて、こちらで計算し直す ----
_UNIT_ALT = r"(?:万円|円|名|人|学科|学部|つ|個|件|校)"
_UNIT = _UNIT_ALT + "?"
_EXPR_RE = re.compile(rf"((?:\d[\d,，]*\s*{_UNIT}\s*[+\-－−ー×*÷/]\s*)+\d[\d,，]*\s*{_UNIT})\s*[=＝]")
_SUMMARY_NAME = re.compile(r"合計|総額|総計|差|計算|結果|答え|合わせ|足す|引く")


def exprs_in_text(text: str, limit: int = 3) -> list:
    """文の中の「A ＋ B ＝ …」を探し、左辺をこちらで計算した [(式, 結果)]。"""
    out = []
    for m in _EXPR_RE.finditer(text or ""):
        e = re.sub(_UNIT_ALT, "", m.group(1))
        e = e.replace("－", "-").replace("−", "-").replace("ー", "-").strip()
        try:
            out.append((m.group(1).strip(), safe_calc(e)))
        except Exception:
            pass
        if len(out) >= limit:
            break
    return out


def values_in_question(question: str) -> list:
    """質問の括弧の中に「項目 数」が2つ以上書かれていれば [(項目, 数)]（例：（情報デザイン100、メディア情報200、情報理工320））。
    質問が数を与えているときは、資料から書き出させた数より、こちらを使う（取り違え・写し間違いを防ぐ）。"""
    for inner in re.findall(r"[（(]([^）)]{4,120})[）)]", question):
        pairs = []
        for part in re.split(r"[、,，・/／]", inner):
            m = re.fullmatch(r"\s*([^\d\s]{1,20}?)\s*[:：]?\s*(\d[\d,，]*(?:\.\d+)?)\s*(万円|円|名|人|件|校|%|％)?\s*", part)
            if m and not re.search(r"年|月|日", m.group(1) + (m.group(3) or "")):
                v = parse_number(m.group(2))
                if v is not None:
                    pairs.append((m.group(1), v))
        if len(pairs) >= 2:
            return pairs
    return []


def clean_values(values: list) -> list:
    """書き出された数値から、モデルが自分で計算した「合計」「差」などの行と、ほかの数の和・差に等しい数を外す。"""
    vals = [(n, v) for n, v in values if not _SUMMARY_NAME.search(n)]
    nums = [v for _, v in vals]
    if len(nums) >= 3:
        keep = []
        for i, (n, v) in enumerate(vals):
            others = nums[:i] + nums[i + 1:]
            if v == sum(others) or (len(others) == 2 and v == abs(others[0] - others[1])):
                continue
            keep.append((n, v))
        vals = keep
    return vals