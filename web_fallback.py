"""Web補完：検索結果が弱いとき、上位ヒットのページからリンクを1段たどって関連ページを追加取得する。

検索APIは使わず、大学サイトのリンク構造（木）だけを使う。
回答時にネット接続が許される場合のみ CFG.web_fallback = True にする。
"""
from config import CFG
from crawl import extract_links, fetch, make_session
from index import char_bigrams
from preprocess import document_to_chunks


def expand_from_hits(question: str, hits: list, known_urls: set) -> list:
    session = make_session()
    q_grams = set(char_bigrams(question))
    candidates = []
    for chunk, _ in hits:
        if not chunk.source.startswith("http"):
            continue
        got = fetch(session, chunk.source)
        if not got or got[0] != "web":
            continue
        for url in extract_links(got[1], chunk.source):
            if url not in known_urls:
                candidates.append(url)

    # URL文字列と質問の重なりで粗く優先度付け（リンク文言を使う改良の余地あり）
    candidates = sorted(set(candidates), key=lambda u: -len(q_grams & set(char_bigrams(u))))
    new_chunks = []
    for url in candidates[: CFG.web_max_fetch]:
        got = fetch(session, url)
        if got:
            new_chunks.extend(document_to_chunks(url, *got))
            known_urls.add(url)
    return new_chunks
