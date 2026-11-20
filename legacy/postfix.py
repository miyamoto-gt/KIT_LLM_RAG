"""生成文の後処理（追加分）：途中から英語に切り替わった回答を、英語が始まる手前で切る。

資料に日英併記のページがあると、3Bモデルが途中から英文の続きを書くことがある（D02で確認）。
既存の後処理は「。」で文を数えるため、英文は1文扱いのまま最後まで残ってしまう。

方針:
  1. 英単語が8語以上続く箇所を「英語の開始」とみなす
     （「CDIO (Conceive, Design, Implement, Operate)」のような短い英語表記は残す）
  2. その手前で最後の「。」まで戻って切る（文を途中で切らない）
  3. 「。」がなければ最後の「、」で切って「。」で閉じる
  4. どちらもなければ、英語の手前でそのまま切る
"""
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


if __name__ == "__main__":
    d02 = ("プロジェクトデザイン教育とは、学生が自ら考え行動する創造的探究・実践力を育成するための教育です。"
           "これは、知識教育と実技教育で培った能力を統合し、実践知に変える役割を担っています。"
           "具体的には、ユーザーが何を必要としているかを考え、問題を発見し、知識やアイデアを組み合わせて"
           "解決策を創出し、具体化して実験・検証・評価するPBL（Project  The process of project design "
           "education is as follows: Problem identification, problem clarification.")
    print(cut_english(d02))
    print(cut_english("KIT国際交流会館は47号館です。"))                    # 英字が短いので変えない
    print(cut_english("授業では、CDIO (Conceive, Design, Implement, Operate) を取り入れています。"))