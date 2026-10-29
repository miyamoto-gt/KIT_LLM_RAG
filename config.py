"""全体設定。値を変えるときはここだけ触る。"""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---- ① 収集 ----
    start_urls: list = field(default_factory=lambda: ["https://www.kanazawa-it.ac.jp/"])
    allowed_domain: str = "kanazawa-it.ac.jp"   # このドメイン配下だけをたどる
    max_depth: int = 3                           # トップページからのリンク深さ
    max_pages: int = 1500                        # 取得ページ数の上限
    crawl_delay: float = 1.0                     # アクセス間隔（秒）。学校サーバーへの配慮

    # ---- ②③ 前処理・チャンク化 ----
    raw_dir: str = "data/raw"
    chunks_path: str = "data/chunks.jsonl"
    extra_chunk_files: list = field(default_factory=lambda: ["data/kit_campus_map.jsonl"])  # 手作りデータ
    chunk_size: int = 500
    chunk_overlap: int = 100

    # ---- ④ ベクトル索引 ----
    index_dir: str = "data/index"
    embed_model: str = "cl-nagoya/ruri-v3-310m"
    query_prefix: str = "検索クエリ: "            # ruri-v3 の作法
    doc_prefix: str = "検索文書: "
    use_bm25: bool = True                         # False でベクトル検索のみ

    # ---- ⑥ 検索 ----
    top_k: int = 4                                # プロンプトに入れるチャンク数
    low_score: float = 0.80                       # これ未満なら「検索が外れた」とみなす（要調整）

    # ---- Web補完（回答時にネット接続が許される場合のみ True） ----
    web_fallback: bool = False
    web_max_fetch: int = 5                        # 1問あたり追加で取得するページ数の上限

    # ---- ⑦ 生成 ----
    llm_model: str = "sbintuitions/sarashina2.2-3b"
    max_new_tokens: dict = field(default_factory=lambda: {"fact": 48, "calc": 48, "explain": 200})
    #---- ⑧ 評価 ----
    extra_chunk_files: list = field(default_factory=lambda: [
    "data/kit_campus_map.jsonl",
    "data/kit_facts.jsonl",
])


CFG = Config()
