# KIT RAG（sarashina2.2-3b）

金沢工業大学（KIT）に関する質問に、大学サイトの情報を検索（RAG）して `sbintuitions/sarashina2.2-3b` で答える。

- 評価セット100問（`eval_set.json`）：**正解率 100%**
- 手作りデータを入れていない分野の10問（`add_question.json`）：**正解率 90%**

##アーキテクチャー

<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/9a32e13c-4a85-4674-89ce-f049a9879694" />

## 全体の流れ

```
━━━━━━ オフライン（索引づくり。済んでいれば不要） ━━━━━━
crawl.py           大学サイトをリンクで巡回して HTML / PDF を保存（Colab で約1050ページ）
preprocess.py      本文抽出（表は Markdown）→ 500字前後に分割 → 文字化けチャンクは除外
index.py           ruri-v3 で埋め込み → FAISS 索引（data/index/）
  ＋ clean_index.py      作成済みの索引から文字化けチャンクを除く
  ＋ reextract_pdfs.py   化けていた PDF を PyMuPDF で読み直す
  ＋ crawl_more.py       指定した分野の、まだ索引に無いページだけを追加で取得
  ＋ add_chunks.py       上記や手作りデータを、全体を作り直さずに索引へ追加

━━━━━━ オンライン（質問ごと） ━━━━━━
pipeline.py  answer(質問)
  router.py      キーワードで 事実（fact）／説明（explain）／計算（calc）に振り分け
  index.py       ベクトル検索＋キーワード検索（BM25）を順位統合
                 → 古い年の資料ほど点数を下げる → 同じページは1件 → 前後のチャンクを連結
                 → 手作りデータ（キャンパスマップ・事実データ）を最大2件優先
  prompts.py     ベースモデル向けの「お手本の続きを書かせる」プロンプト
  generator.py   sarashina2.2-3b で生成（CUDA / Mac の MPS / CPU を自動で選ぶ）
  tools.py       計算は式だけをモデルに書かせ、Python で安全に計算
  postprocess    事実は1行、説明は最大3文（箇条書きは1文にまとめる）
```

## ファイル

| ファイル | 役割 |
|---|---|
| `config.py` | 設定（クロール範囲、チャンク長、モデル名、検索件数など） |
| `schema.py` | 全情報源で共通のチャンク形式（id / text / source / title / breadcrumb / type） |
| `crawl.py` | ① 収集。トップから幅優先でリンクをたどる |
| `crawl_more.py` | 追加収集。出発点を指定し、新しいページだけをチャンクにする |
| `preprocess.py` | ②③ 本文抽出・PDF 読み取り・分割・文字化け判定（`is_readable`） |
| `index.py` | ④ 索引作成と ⑥ 検索（`Retriever`） |
| `clean_index.py` | 索引から文字化けチャンクを除く |
| `reextract_pdfs.py` | 化けていた PDF を取り直して PyMuPDF で読む |
| `add_chunks.py` | JSONL のチャンクを索引に追加・入れ替え |
| `router.py` | 質問の種類の振り分け |
| `prompts.py` | プロンプトのお手本 |
| `generator.py` | sarashina の読み込みと生成 |
| `tools.py` | 計算ツール（四則演算だけを許す `safe_calc`） |
| `pipeline.py` | 質問1つに答える中心部品。コマンドとしても使える |
| `evaluate.py` | 評価セットを解いて自動採点（キーワード一致、外部 API 不要） |
| `run_eval.py` | 本番用。質問 JSON の全問に答えて `answers.json` を作る |
| `eval_set.json` | 評価用100問（正解キーワード・禁止語つき） |
| `add_question.json` | 手作りデータの無い分野の確認用10問 |
| `colab_run.ipynb` | Colab で索引づくりをするためのノートブック |
| `data/kit_campus_map.jsonl` | キャンパスマップを書き起こした手作りデータ（32件） |
| `data/kit_facts.jsonl` | 学部・学科、学費（大学公式）、会費、学長、図書館フロア、オナーズなどの事実データ |
| `data/pdf_fixed.jsonl` | 読み直した PDF のチャンク |
| `data/more_chunks.jsonl` | 追加クロールで取ったチャンク |
| `data/index/` | 索引（`faiss.index` と `chunks.jsonl`）。Git には入れず Google ドライブに控える |

チャンクの種類（`type`）：`web` / `pdf`（クロール）、`map`（キャンパスマップ）、`manual`（事実データ）。
`map` と `manual` は検索で優先され、古さによる減点を受けない。

## 環境（Mac・Apple Silicon で確認）

- Python 3.12（3.14 では `transformers` が動かなかった）
- `transformers>=4.48,<5`、`sentencepiece`、`protobuf`、`httpx`、`pymupdf` が必要（`requirements.txt`）

```bash
uv venv -p 3.12 .venv
uv pip install -r requirements.txt
```

実行はどちらか。
```bash
uv run python pipeline.py "質問"            # uv で実行
source .venv/bin/activate && python pipeline.py "質問"   # .venv を有効にして実行
```

索引（`data/index/`）は Google ドライブの `kit_rag_data/index/` から取ってきて置く。
初回はモデル（sarashina 約6〜7GB、ruri 約1.3GB）のダウンロードに時間がかかる。

## 使い方

```bash
# 質問する（複数可。--raw でモデルの生出力も表示）
uv run python pipeline.py "図書館は何号館にありますか？"
uv run python pipeline.py --raw "図書館の6階には何がありますか？"
uv run python pipeline.py              # 対話モード（空行で終了）

# 評価
uv run python evaluate.py                          # 100問
uv run python evaluate.py --set add_question.json  # 確認用10問
uv run python evaluate.py --ids A09 E06            # 指定した問題だけ
#  → 最後に正解率、間違えた問題は eval_failures.jsonl、全問の詳細は eval_report.json

# 本番
uv run python run_eval.py questions.json           # → answers.json / answers_log.json
```

`run_eval.py` 冒頭の `TODO`（質問・ID・答えのキー名、送信先 URL）を採点の形式に合わせて使う。

## データを足すとき

```bash
# 手作りデータ（同じ id は入れ替え）
uv run python add_chunks.py data/kit_facts.jsonl data/kit_campus_map.jsonl

# 分野を指定して追加クロール → 追加
uv run python crawl_more.py https://www.kanazawa-it.ac.jp/kitnews/ --max-pages 150 --depth 2
uv run python add_chunks.py data/more_chunks.jsonl --replace-sources --exclude mondai_
```

手作りデータは、種類（`type`）を書かずに `id` / `title` / `source` / `text` だけを書く（`source` が「キャンパスマップ」なら `map`、それ以外は `manual` として扱われる）。
「〇〇は〇号館の〇階にある」「〇号館の〇階には〇〇がある」のように、聞かれ方に合わせた文を入れると検索で当たりやすい。

## 主な設定（`config.py`、書かなければ既定値）

| 項目 | 既定 | 意味 |
|---|---|---|
| `top_k` | 4 | プロンプトに入れる資料の数 |
| `use_bm25` | True | キーワード検索も併用するか |
| `max_per_source` | 1 | 同じページから取るチャンク数 |
| `expand_neighbors` | 1 | 前後に何チャンクつなげるか |
| `priority_top` | 2 | 手作りデータを最大何件優先するか |
| `priority_margin` | 0.05 | 手作りデータを優先する類似度の差 |
| `current_year` / `recency_decay` | 2026 / 0.9 | 1年古くなるごとに検索の点数を何倍にするか（1.0で無効） |
| `web_fallback` | False | 回答時に大学サイトを追加取得するか |

## つまずいたところと対処

| 症状 | 原因と対処 |
|---|---|
| `tiktoken is required` | `sentencepiece` が無い。Python 3.12 ＋ `transformers<5` で環境を作り直す |
| `No module named 'httpx'` / `protobuf` | `uv pip install httpx protobuf` |
| `segmentation fault`（埋め込み中） | Mac で faiss と torch がぶつかる。`add_chunks.py` は埋め込み後に faiss を読む作り。`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` も有効 |
| 途中から全問が空の答え | GPU（MPS）のメモリ詰まり。`evaluate.py` / `run_eval.py` は毎問メモリを片付け、失敗時は1回やり直す。ほかのアプリを閉じる |
| VS Code で「File is read-only」 | 受け取ったファイルが読み取り専用。`chmod u+w *.py *.json *.md data/*.jsonl` |
| 答えが資料の続き（「### 資料5」など）になる | ベースモデルは指示より「続き」を書く。`prompts.py` のお手本形式で対処済み |
| PDF の中身が化ける | フォントに文字の対応表が無い PDF。`reextract_pdfs.py`（PyMuPDF）で読み直し |

## 既知の弱点

- ニュースなど過去の出来事：記事の取りこぼしと、古さによる減点で弱い
- 範囲の広い質問（「大学の特徴」など）：モデルの知識が混ざることがある
- 評価セットは索引に入れた情報から作っているため、本番の問題では正解率が下がる可能性がある
