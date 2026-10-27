# KIT RAG（sarashina2.2-3b）土台

## アーキテクチャ

```
━━━━━━━━━━ オフライン（事前に1回） ━━━━━━━━━━
crawl.py       ① 収集      トップからリンクを幅優先でたどる（木構造を manifest に記録）
preprocess.py  ②③ 前処理   本文抽出・表はMarkdown・PDF対応 → 500字前後のチャンク
                             ＋ data/kit_campus_map.jsonl など手作りデータを合流
index.py       ④ 索引      ruri-v3 で埋め込み → FAISS（＋BM25）

━━━━━━━━━━ オンライン（質問ごと） ━━━━━━━━━━
run_eval.py    ⑤ 質問JSONを読み込み、全問ループ
pipeline.py      answer(question) が唯一の入口
  router.py      振り分け   fact / calc / explain
  index.py       検索       ベクトル＋BM25 を順位統合（RRF）
  web_fallback.py  Web補完  類似度が低いとき、上位ヒットのページからリンクを1段たどって追加取得（任意）
  prompts.py     プロンプト  質問タイプごとに出力指示を変える
  generator.py   生成       sarashina2.2-3b
  tools.py       計算       <calc>式</calc> を Python で安全に計算
               ⑧ 後処理 → answers.json
```

すべての情報源は `schema.Chunk`（id / text / source / title / breadcrumb / type）に揃える。

## 実行順（Colab）

`colab_run.ipynb` を上から実行。手動なら：

```bash
pip install -r requirements.txt
python crawl.py        # 数十分（1秒間隔でアクセス）
python preprocess.py
python index.py
python pipeline.py "図書館は何号館にありますか？"
python run_eval.py questions.json
```

## 差し替えポイント（あとで強化する場所）

| やりたいこと | 触るファイル |
|---|---|
| 振り分けを分類器に（TF-IDF＋ロジスティック回帰） | `router.py` の `classify` |
| 埋め込みを学内データで微調整（ドメイン特化・検索側） | `config.embed_model` を微調整済みモデルに |
| RAFT（ドメイン特化・LLM側） | `config.llm_model` を LoRA 適用済みモデルに |
| ベクトル検索のみで比較 | `config.use_bm25 = False` |
| 回答時のWeb補完を有効化 | `config.web_fallback = True` |
| 採点APIの形式に合わせる | `run_eval.py` の TODO |

## 調整が必要な値

- `low_score`：Web補完を発動する類似度のしきい値。自作質問で「正解時／不正解時の top_score」を見て決める。
- `max_new_tokens`：短答が長すぎるなら fact を減らす。
- `chunk_size`：検索が当たらないなら 300〜800 で比較。
