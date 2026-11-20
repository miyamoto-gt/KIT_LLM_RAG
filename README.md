# KIT RAG

金沢工業大学（KIT）に関する質問に、大学サイトの情報を検索（RAG）して答える。

## 提出版（API 版）— 提出に使うのは `submission/` だけ

本番のサンドボックスでは torch / transformers が使えないため、LLM（`sarashina2.2-3b`）と埋め込みは
運営の OpenAI 互換 API で呼ぶ。提出物は `submission/`（`main.py`・`kitrag/`・`knowledge/`・`requirements.txt`）だけで、
直下の旧版（torch 版）は `legacy/` に移した（データづくりと旧評価のために動く状態で残している）。

```
KIT_LLM_RAG/
├── submission/        # 提出するフォルダ（main.py, kitrag/, knowledge/）
├── submit.py          # 運営の提出クライアント（rag-starter-kit からコピー。標準ライブラリだけで動く）
├── legacy/            # 旧版（torch 版）のモジュール・main.py・make_zip.sh・requirements.txt
├── build/             # データづくり（crawl・preprocess など）
├── evaluation/        # 旧版の評価。質問セットは evaluation/sets/
├── data/              # 索引と手作りデータ
├── outputs/           # 生成物（questions.csv・submission.csv もここに出す。Git に入れない）
├── docs/
└── requirements.txt   # 手元の開発用（openai, numpy, rank-bm25, faiss-cpu, pandas）
```

### 手元で試す・提出する（すべて直下で実行）

```bash
pip install -r requirements.txt
python submit.py login --url http://192.168.38.239:3000
python submit.py questions -o outputs/questions.csv
python submit.py run submission --questions outputs/questions.csv -o outputs/submission.csv
python submit.py submit outputs/submission.csv --code submission --name "<名前>"
python submit.py status
```

`submit.py` が `submission/` を zip にまとめて送るので、`make_zip.sh` はもう使わない（`legacy/` に保管）。

### 資料を作り直すとき

```bash
python build/export_knowledge.py     # → submission/knowledge/（旧版に依存しないので PYTHONPATH は不要）
```

> ※ 2026-11-19 時点で `build/export_knowledge.py` はまだリポジトリに無い。作成したらこのコマンドで動く想定。

## 旧版（legacy）の build/ と evaluation/ を動かす

旧版のモジュールは `from config import CFG` のように互いを直接 import している。import 文は書き換えず、
**直下から `PYTHONPATH=legacy` を付けて `python -m` で実行する**（`data/index` などの相対パスは直下基準のまま）。

```bash
PYTHONPATH=legacy python -m build.crawl_more https://www.kanazawa-it.ac.jp/kitnews/ --max-pages 150 --depth 2
PYTHONPATH=legacy python -m build.add_chunks data/more_chunks.jsonl --replace-sources --exclude mondai_
PYTHONPATH=legacy python -m evaluation.rag_check --set evaluation/sets/public_set.json
PYTHONPATH=legacy python -m evaluation.evaluate
PYTHONPATH=legacy python -m pipeline "図書館は何号館にありますか？"   # 旧版で1問答える
PYTHONPATH=legacy python legacy/main.py                              # 旧版の提出入口
```

旧版の依存（torch 等）は `legacy/requirements.txt`。以下は旧版の説明（当時のまま。コマンドは上の形に読み替える）。

---

## 旧版（legacy）：KIT RAG（sarashina2.2-3b）

金沢工業大学（KIT）に関する質問に、大学サイトの情報を検索（RAG）して `sbintuitions/sarashina2.2-3b` で答える。

- 評価セット100問（`evaluation/sets/eval_set.json`）：**正解率 100%**
- 手作りデータを入れていない分野の10問（`evaluation/sets/add_question.json`）：**正解率 90%**

### アーキテクチャー

<img width="500" height="500" alt="Image" src="https://github.com/user-attachments/assets/6e6814c5-5bd9-4d3b-89d1-1b8e1d2ed570" />

### 全体の流れ

```
━━━━━━ オフライン（索引づくり。済んでいれば不要） ━━━━━━
build/crawl.py     大学サイトをリンクで巡回して HTML / PDF を保存（Colab で約1050ページ）
build/preprocess.py 本文抽出（表は Markdown）→ 500字前後に分割 → 文字化けチャンクは除外
index.py           ruri-v3 で埋め込み → FAISS 索引（data/index/）
  ＋ build/clean_index.py      作成済みの索引から文字化けチャンクを除く
  ＋ build/reextract_pdfs.py   化けていた PDF を PyMuPDF で読み直す
  ＋ build/crawl_more.py       指定した分野の、まだ索引に無いページだけを追加で取得
  ＋ build/add_chunks.py       上記や手作りデータを、全体を作り直さずに索引へ追加

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

### フォルダ構成

```
KIT_LLM_RAG/
├── legacy/main.py     # 旧版の提出の入口（questions.csv → submission.csv）
├── legacy/            # pipeline.py config.py schema.py router.py prompts.py postfix.py
│                      # tools.py generator.py index.py web_fallback.py（回答時に使うモジュール）
├── build/             # 索引づくり（オフライン）
├── evaluation/        # 評価・診断・ローカル実行。質問セットは evaluation/sets/
├── data/              # 索引と手作りデータ（legacy/config.py が参照）
├── outputs/           # 生成物（Git・zip に入れない）
└── docs/
```

`build/` と `evaluation/` のスクリプトは、**すべて直下から `PYTHONPATH=legacy python -m` で実行する**（例：`PYTHONPATH=legacy python -m build.add_chunks ...`）。
旧版の提出用 zip は `bash legacy/make_zip.sh` で作っていた（現在は `submit.py` が zip を作るので不要）。

### ファイル

| ファイル | 役割 |
|---|---|
| `config.py` | 設定（クロール範囲、チャンク長、モデル名、検索件数など） |
| `schema.py` | 全情報源で共通のチャンク形式（id / text / source / title / breadcrumb / type） |
| `build/crawl.py` | ① 収集。トップから幅優先でリンクをたどる |
| `build/crawl_more.py` | 追加収集。出発点を指定し、新しいページだけをチャンクにする |
| `build/preprocess.py` | ②③ 本文抽出・PDF 読み取り・分割・文字化け判定（`is_readable`） |
| `index.py` | ④ 索引作成と ⑥ 検索（`Retriever`） |
| `build/clean_index.py` | 索引から文字化けチャンクを除く |
| `build/reextract_pdfs.py` | 化けていた PDF を取り直して PyMuPDF で読む |
| `build/add_chunks.py` | JSONL のチャンクを索引に追加・入れ替え |
| `router.py` | 質問の種類の振り分け |
| `prompts.py` | プロンプトのお手本 |
| `generator.py` | sarashina の読み込みと生成 |
| `tools.py` | 計算ツール（四則演算だけを許す `safe_calc`） |
| `pipeline.py` | 質問1つに答える中心部品。コマンドとしても使える |
| `evaluation/evaluate.py` | 評価セットを解いて自動採点（キーワード一致、外部 API 不要） |
| `evaluation/run_eval.py` | 質問 JSON の全問に答えて `outputs/answers.json` を作る |
| `evaluation/sets/eval_set.json` | 評価用100問（正解キーワード・禁止語つき） |
| `evaluation/sets/add_question.json` | 手作りデータの無い分野の確認用10問 |
| `build/colab_run.ipynb` | Colab で索引づくりをするためのノートブック |
| `data/kit_campus_map.jsonl` | キャンパスマップを書き起こした手作りデータ（32件） |
| `data/kit_facts.jsonl` | 学部・学科、学費（大学公式）、会費、学長、図書館フロア、オナーズなどの事実データ |
| `data/pdf_fixed.jsonl` | 読み直した PDF のチャンク |
| `data/more_chunks.jsonl` | 追加クロールで取ったチャンク |
| `data/index/` | 索引（`faiss.index` と `chunks.jsonl`）。Git には入れず Google ドライブに控える |

チャンクの種類（`type`）：`web` / `pdf`（クロール）、`map`（キャンパスマップ）、`manual`（事実データ）。
`map` と `manual` は検索で優先され、古さによる減点を受けない。

### 環境（Mac・Apple Silicon で確認）

- Python 3.12（3.14 では `transformers` が動かなかった）
- `transformers>=4.48,<5`、`sentencepiece`、`protobuf`、`httpx`、`pymupdf` が必要（`legacy/requirements.txt`）

```bash
uv venv -p 3.12 .venv
uv pip install -r legacy/requirements.txt
```

実行はどちらか。
```bash
PYTHONPATH=legacy PYTHONPATH=legacy uv run python -m pipeline "質問"            # uv で実行
source .venv/bin/activate && PYTHONPATH=legacy python -m pipeline "質問"   # .venv を有効にして実行
```

索引（`data/index/`）は Google ドライブの `kit_rag_data/index/` から取ってきて置く。
初回はモデル（sarashina 約6〜7GB、ruri 約1.3GB）のダウンロードに時間がかかる。

### 使い方

```bash
# 質問する（複数可。--raw でモデルの生出力も表示）
PYTHONPATH=legacy PYTHONPATH=legacy uv run python -m pipeline "図書館は何号館にありますか？"
PYTHONPATH=legacy PYTHONPATH=legacy uv run python -m pipeline --raw "図書館の6階には何がありますか？"
PYTHONPATH=legacy PYTHONPATH=legacy uv run python -m pipeline              # 対話モード（空行で終了）

# 評価
PYTHONPATH=legacy uv run python -m evaluation.evaluate                # 100問
PYTHONPATH=legacy uv run python -m evaluation.evaluate --set evaluation/sets/add_question.json  # 確認用10問
PYTHONPATH=legacy uv run python -m evaluation.evaluate --ids A09 E06  # 指定した問題だけ
#  → 最後に正解率、間違えた問題は outputs/eval_failures.jsonl、全問の詳細は outputs/eval_report.json

# 本番（提出物と同じ入口）
PYTHONPATH=legacy uv run python legacy/main.py                              # questions.csv → submission.csv
PYTHONPATH=legacy uv run python -m evaluation.run_eval evaluation/sets/questions.json   # → outputs/answers.json / answers_log.json
```

`evaluation/run_eval.py` 冒頭の `TODO`（質問・ID・答えのキー名、送信先 URL）を採点の形式に合わせて使う。

### データを足すとき

```bash
# 手作りデータ（同じ id は入れ替え）
PYTHONPATH=legacy uv run python -m build.add_chunks data/kit_facts.jsonl data/kit_campus_map.jsonl

# 分野を指定して追加クロール → 追加
PYTHONPATH=legacy uv run python -m build.crawl_more https://www.kanazawa-it.ac.jp/kitnews/ --max-pages 150 --depth 2
PYTHONPATH=legacy uv run python -m build.add_chunks data/more_chunks.jsonl --replace-sources --exclude mondai_
```

手作りデータは、種類（`type`）を書かずに `id` / `title` / `source` / `text` だけを書く（`source` が「キャンパスマップ」なら `map`、それ以外は `manual` として扱われる）。
「〇〇は〇号館の〇階にある」「〇号館の〇階には〇〇がある」のように、聞かれ方に合わせた文を入れると検索で当たりやすい。

### 主な設定（`config.py`、書かなければ既定値）

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

### つまずいたところと対処

| 症状 | 原因と対処 |
|---|---|
| `tiktoken is required` | `sentencepiece` が無い。Python 3.12 ＋ `transformers<5` で環境を作り直す |
| `No module named 'httpx'` / `protobuf` | `uv pip install httpx protobuf` |
| `segmentation fault`（埋め込み中） | Mac で faiss と torch がぶつかる。`build/add_chunks.py` は埋め込み後に faiss を読む作り。`OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` も有効 |
| 途中から全問が空の答え | GPU（MPS）のメモリ詰まり。`evaluation/evaluate.py` / `evaluation/run_eval.py` は毎問メモリを片付け、失敗時は1回やり直す。ほかのアプリを閉じる |
| VS Code で「File is read-only」 | 受け取ったファイルが読み取り専用。`chmod u+w *.py */*.py *.md evaluation/sets/*.json data/*.jsonl` |
| 答えが資料の続き（「### 資料5」など）になる | ベースモデルは指示より「続き」を書く。`prompts.py` のお手本形式で対処済み |
| PDF の中身が化ける | フォントに文字の対応表が無い PDF。`build/reextract_pdfs.py`（PyMuPDF）で読み直し |

### 進捗（2026-11-13 時点）
 
#### スコア
 
| 提出 | Public 正解率 | Private |
|---|---|---|
| 最初のベースライン | 78 | コード実行エラー |
| 振り分け・生成の改善後 | 87.9 | コード実行エラー |
| 同じ回答CSVで再提出 | 86.5 | コード実行エラー |
 
- 同じ回答CSVでも Public 正解率は揺れる（87.9 / 86.5）。**±1.5 程度は誤差**として扱い、改善の判断は手元の評価セットで行う
#### 手元の評価（キーワード判定）
 
| 評価セット | 中身 | 結果 |
|---|---|---|
| `eval_set.json` | 索引の情報から作った100問。壊していないかの確認 | 説明・計算問題（D・E）は全問正解 |
| `public_set.json` | 公開問題31問に正解キーワードを付けたもの。**測定専用** | 27/30（q068 は正解不明で対象外） |
| `paraphrase_set.json` | 自作の言い換え25問。新しい言い回しへの強さを測る | 24/25 |
 
#### 入れた改善（すべて汎用の仕組み）
 
| 部分 | 変更 |
|---|---|
| 検索（`config.py`） | 同じページから取るチャンク数を 1 → 2（大きい PDF から2件取れる） |
| 振り分け（`router.py`） | 単語を足す方式をやめ、質問の形で判定。計算＝「計算の言葉＋数を問う言葉」、「〜より高い／多い」も計算、依頼形で短答の疑問詞が無ければ説明、列挙は事実。答えの部分の数を数える `plan()` |
| 生成（`pipeline.py`・`prompts.py`・`tools.py`） | 部分の数に合わせて出力の上限を広げる。計算結果が答えに無ければ添える。複数部分の計算は複数文で答える。式を最大3つ計算。お手本は手作りデータから作成 |
| 提出（`main.py`） | 知らない引数を無視・質問CSVを中身で探す・モデルが読めなくても「不明」でCSVを書き、必ず終了コード0で終わる |
| 評価（`evaluation/stage_check.py`） | データ → 検索 → 生成 → 計算ツール → 確信度 を段階ごとに診断。検索設定を変えて比べられる |
 
### 既知の弱点

- ニュースなど過去の出来事：記事の取りこぼしと、古さによる減点で弱い
- 範囲の広い質問（「大学の特徴」など）：モデルの知識が混ざることがある
- 評価セットは索引に入れた情報から作っているため、本番の問題では正解率が下がる可能性がある
