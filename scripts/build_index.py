"""Сборка векторного индекса FAISS из базы знаний.

Пайплайн (см. конспект «Генерация эмбеддингов и векторные базы»):
  knowledge_base/**/*.md
    → парсинг frontmatter (метаданные для цитат и фильтрации)
    → нарезка тела на чанки с overlap (RecursiveCharacterTextSplitter,
      длина считается в ТОКЕНАХ модели)
    → эмбеддинг Qwen3-Embedding-0.6B (нормировка → косинус)
    → FAISS (IndexFlatIP, точный поиск — корпус маленький)
    → сохранение в faiss_index/

Запуск:
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    EMBED_REVISION,
    INDEX_DIR,
    KB_DIR,
    ROOT,
    get_embeddings,
)


def parse_doc(path: Path) -> tuple[str, dict]:
    """Возвращает (тело, метаданные) markdown-документа с YAML-frontmatter."""
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = text
    if text.startswith("---"):
        # ---\n<frontmatter>\n---\n<body>
        _, frontmatter, body = text.split("---", 2)
        meta = yaml.safe_load(frontmatter) or {}
    meta = {k: str(v) for k, v in meta.items()}
    meta["source"] = path.relative_to(ROOT).as_posix()
    return body.strip(), meta


def load_chunks(
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[list[Document], list[Path]]:
    """Читает базу знаний и режет её на чанки-документы с метаданными."""
    chunks: list[Document] = []
    files = sorted(KB_DIR.rglob("*.md"))
    for path in files:
        body, meta = parse_doc(path)
        for i, piece in enumerate(splitter.split_text(body)):
            chunks.append(Document(page_content=piece, metadata={**meta, "chunk_index": i}))
    return chunks, files


def main() -> None:
    t0 = time.perf_counter()

    # Сплиттер: длина в токенах модели (а не в символах) — контролируем ту же
    # «линейку», которой меряется лимит эмбеддера.
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL, revision=EMBED_REVISION)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
    )

    chunks, files = load_chunks(splitter)
    if not chunks:
        print("Ошибка: в knowledge_base/ не найдено документов.")
        sys.exit(1)

    by_lang: dict[str, int] = {}
    for c in chunks:
        by_lang[c.metadata.get("lang", "?")] = by_lang.get(c.metadata.get("lang", "?"), 0) + 1

    print(f"Документов: {len(files)} | чанков: {len(chunks)} | по языку: {by_lang}")
    print(f"Эмбеддер: {EMBED_MODEL} (rev={EMBED_REVISION}) — загрузка и кодирование…")

    embeddings = get_embeddings(show_progress=True)
    store = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # косинус по нормированным
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))

    dt = time.perf_counter() - t0
    size_mb = sum(f.stat().st_size for f in INDEX_DIR.glob("*")) / (1024 * 1024)
    print("-" * 60)
    print(f"OK: индекс сохранён в {INDEX_DIR.relative_to(ROOT).as_posix()}/")
    print(f"  векторов: {store.index.ntotal} | размерность: {store.index.d}")
    print(f"  тип индекса: {type(store.index).__name__} (точный, inner product)")
    print(f"  размер на диске: {size_mb:.1f} МБ | время сборки: {dt:.1f} c")


if __name__ == "__main__":
    main()
