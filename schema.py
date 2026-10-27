"""全情報源（Web・PDF・マップ書き起こし）で共通のチャンク形式。"""
import json
from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    id: str
    text: str            # 埋め込み・プロンプトに使う本文（先頭にタイトル等を含める）
    source: str = ""     # URL など
    title: str = ""
    breadcrumb: str = ""
    type: str = "web"    # web / pdf / map

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]


def save_chunks(chunks: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
