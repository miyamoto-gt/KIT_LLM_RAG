# KIT RAG（sarashina2.2-3b）土台

## アーキテクチャ

https://github.com/miyamoto-gt/KIT_LLM_RAG/issues/1#issue-6081290940

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
