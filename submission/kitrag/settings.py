"""設定。値を変えるときはここだけ触る（環境変数で上書きできるものは KITRAG_ を付ける）。"""
import os
import time
from pathlib import Path

START = time.time()                                    # 実行開始（main.py が最初に読み込む）

ROOT = Path(__file__).resolve().parent.parent          # submission/
KNOWLEDGE_DIR = ROOT / "knowledge"
HANDMADE_DIR = "_handmade"                             # knowledge/_handmade/ は手作りデータ（検索で優先）

# ---- API（サンドボックスが渡す環境変数）----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
API_KEY = os.environ.get("OPENAI_API_KEY", "sandbox")
LLM_MODEL = os.environ.get("LLM_MODEL", "sarashina2.2-3b")
# 埋め込みは自由に選べる（Hugging Face のリポジトリ名・3GB 以下。サーバーが初回にダウンロードする）。
# 本番の環境変数 EMBEDDING_MODEL は既定の sarashina なので、使うモデルはここ（KITRAG_EMBED_MODEL）で決める。
EMBEDDING_MODEL = os.environ.get("KITRAG_EMBED_MODEL", "sarashina-embedding-v2-1b")
_PREFIXES = {                                          # モデルカードに書かれた接頭辞（質問, 資料）
    "sarashina-embedding-v2-1b": ("query: ", "text: "),
    "cl-nagoya/ruri-v3-310m": ("検索クエリ: ", "検索文書: "),
    "cl-nagoya/ruri-v3-130m": ("検索クエリ: ", "検索文書: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-m3": ("", ""),
}
QUERY_PREFIX, DOC_PREFIX = _PREFIXES.get(EMBEDDING_MODEL, ("query: ", "text: "))

# ---- 時間（制限 30 分）----
TIME_LIMIT = float(os.environ.get("KITRAG_TIME_LIMIT", 25 * 60))   # ここで打ち切り、残りは資料の一文で答えて書き出す（制限30分）
EMBED_TIME_LIMIT = float(os.environ.get("KITRAG_EMBED_TIME_LIMIT", 12 * 60))  # 資料の埋め込みにかけてよい時間
WORKERS = int(os.environ.get("KITRAG_WORKERS", 4))       # 並列に答える質問数
EMBED_WORKERS = int(os.environ.get("KITRAG_EMBED_WORKERS", 4))
EMBED_BATCH = int(os.environ.get("KITRAG_EMBED_BATCH", 32))   # 1回の API に送るチャンク数
REQUEST_TIMEOUT = float(os.environ.get("KITRAG_TIMEOUT", 180))   # 1回の API の待ち時間（混んでいると順番待ちが長い。締め切りが近ければ短く）

# ---- チャンク ----
CHUNK_CHARS, CHUNK_OVERLAP = 500, 100

# ---- 検索 ----
TOP_K = 4                    # 生成に渡すチャンク数
POOL = 40                    # ベクトル・BM25 それぞれで見る候補数
MAX_PER_SOURCE = 2           # 同じページから取る数
EXPAND_NEIGHBORS = 1         # 前後に何チャンクつなげるか
PRIORITY_TOP = 2             # 手作りデータを最大何件優先するか
PRIORITY_MARGIN = 0.05       # 一番近いチャンクとの類似度の差がこれ以内なら手作りデータを優先
KEEP_TOP1 = os.environ.get("KITRAG_KEEP_TOP1", "0") == "1"   # 各検索の1位を必ず残す（未検証なので既定は切る）
CURRENT_YEAR, RECENCY_DECAY, MAX_AGE = 2026, 0.9, 5

# ---- 生成 ----
MAX_NEW_TOKENS = {"fact": 64, "calc": 64, "explain": 200}
NARROW = os.environ.get("KITRAG_NARROW", "1") == "1"   # 資料を質問に関係する文だけに絞る
NARROW_KEEP, NARROW_WIN = 8, 1
FALLBACK = "不明"
MAX_ANSWER_CHARS = 2000


def remaining() -> float:
    """締め切り（TIME_LIMIT）までの残り秒数。"""
    return TIME_LIMIT - (time.time() - START)
