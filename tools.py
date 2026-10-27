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


def run_calc_tags(text: str):
    """最初の <calc>…</calc> を計算して (式, 結果) を返す。無ければ None。"""
    m = re.search(r"<calc>(.*?)(?:</calc>|$)", text, re.S | re.M)
    expr = m.group(1) if m else (text.strip().splitlines() or [""])[0]  # タグ無しなら1行目を試す
    try:
        return expr.strip(), safe_calc(expr)
    except Exception:
        return None
