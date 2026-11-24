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
# カンマで区切って複数書くと、モデルごとの順位と BM25 を RRF でまとめる（例: "sarashina-embedding-v2-1b,cl-nagoya/ruri-v3-310m"）
EMBED_MODELS = [m.strip() for m in os.environ.get("KITRAG_EMBED_MODEL", "sarashina-embedding-v2-1b").split(",") if m.strip()]
EMBEDDING_MODEL = EMBED_MODELS[0]                      # 手作りデータの優先の判定に使うモデル
_PREFIXES = {                                          # モデルカードに書かれた接頭辞（質問, 資料）
    "sarashina-embedding-v2-1b": ("query: ", "text: "),
    "cl-nagoya/ruri-v3-310m": ("検索クエリ: ", "検索文書: "),
    "cl-nagoya/ruri-v3-130m": ("検索クエリ: ", "検索文書: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-m3": ("", ""),
}


def prefixes(model):
    """(質問の接頭辞, 資料の接頭辞)"""
    return _PREFIXES.get(model, ("query: ", "text: "))


QUERY_PREFIX, DOC_PREFIX = prefixes(EMBEDDING_MODEL)

# ---- 時間（制限 30 分）----
TIME_LIMIT = float(os.environ.get("KITRAG_TIME_LIMIT", 25 * 60))   # ここで打ち切り、残りは資料の一文で答えて書き出す（制限30分）
EMBED_TIME_LIMIT = float(os.environ.get("KITRAG_EMBED_TIME_LIMIT", 8 * 60))   # 資料の埋め込みの上限。ふだん約5分。超えたら BM25 に任せて生成の時間を守る（生成のほうが1分あたりの価値が大きい）
WORKERS = int(os.environ.get("KITRAG_WORKERS", 6))       # 並列に答える質問数（11/24 実測：4→6で速く、8はタイムアウトが増える）
PHRASE_BOOST = os.environ.get("KITRAG_PHRASE_BOOST", "0") == "1"   # 質問の（専願型）「…」を含む資料を前に出す（11/24：BM25だけの確認で効果なし→既定は使わない）
RETRY_MIN_LEFT = int(os.environ.get("KITRAG_RETRY_MIN_LEFT", 150))   # 2周目（生成に失敗した問題の答え直し）を始める残り秒数の下限
EMBED_WORKERS = int(os.environ.get("KITRAG_EMBED_WORKERS", 1))   # 資料の埋め込みは1つずつ順番に送る（同時に送ると 502 になりやすく、速さもほぼ同じ）
EMBED_BATCH = int(os.environ.get("KITRAG_EMBED_BATCH", 32))   # 1回の API に送るチャンク数
REQUEST_TIMEOUT = float(os.environ.get("KITRAG_TIMEOUT", 180))   # 1回の API の待ち時間（混んでいると順番待ちが長い。締め切りが近ければ短く）

# ---- チャンク ----
CHUNK_CHARS, CHUNK_OVERLAP = 500, 100

# ---- 検索 ----
TOP_K = 4                    # 生成に渡すチャンク数（説明・計算・複数の部分を聞く質問）
TOP_K_FACT = int(os.environ.get("KITRAG_TOPK_FACT", TOP_K))   # 答えが1つの事実の質問に渡すチャンク数（少ないほど速く、気が散りにくい）
POOL = 40                    # ベクトル・BM25 それぞれで見る候補数
MAX_PER_SOURCE = 2           # 同じページから取る数
EXPAND_NEIGHBORS = 1         # 前後に何チャンクつなげるか
PRIORITY_TOP = 2             # 手作りデータを最大何件優先するか
PRIORITY_MARGIN = 0.05       # 一番近いチャンクとの類似度の差がこれ以内なら手作りデータを優先
KEEP_TOP1 = os.environ.get("KITRAG_KEEP_TOP1", "0") == "1"   # 各検索の1位を必ず残す（未検証なので既定は切る）
CURRENT_YEAR, RECENCY_DECAY, MAX_AGE = 2026, 0.9, 5
# 過去のことを聞く質問（開学時・沿革・昭和や2019年以前の年）は、新しい資料を優先しない（古い資料が答えになるため）
import re as _re
PAST_Q = _re.compile(r"開学|創立|設立当初|設置された|前身|沿革|歴史|当時|昭和|平成|19\d\d年|20[01]\d年")

# ---- 生成 ----
MAX_NEW_TOKENS = {"fact": 64, "calc": 64, "explain": 200}
NARROW = os.environ.get("KITRAG_NARROW", "1") == "1"   # 資料を質問に関係する文だけに絞る
LIST_EXTEND = int(os.environ.get("KITRAG_LIST_EXTEND", 8))   # 絞り込みで、箇条書きの続きの項目を何文まで残すか（0で使わない）
NARROW_KEEP, NARROW_WIN = 8, 1          # 答えが1つの事実の質問で残す文の数
NARROW_KEEP_EXPLAIN = int(os.environ.get("KITRAG_KEEP_EXPLAIN", 8))    # 説明・計算の質問で残す文の数（12 も試したが、良くも悪くもならなかった：11/21）
DECOMPOSE = os.environ.get("KITRAG_DECOMPOSE", "1") == "1"   # 「AとBは、それぞれ〜」を分けて答える
REPAIR = os.environ.get("KITRAG_REPAIR", "1") == "1"         # 答えの形（号館・人数など）が無ければ1回だけ直させる
FALLBACK = "不明"
MAX_ANSWER_CHARS = 2000


def remaining() -> float:
    """締め切り（TIME_LIMIT）までの残り秒数。"""
    return TIME_LIMIT - (time.time() - START)


def top_k_for(p) -> int:
    """振り分けの結果（router.plan）に合わせた、生成に渡すチャンク数。"""
    return TOP_K_FACT if p["qtype"] == "fact" and p["parts"] <= 1 else TOP_K