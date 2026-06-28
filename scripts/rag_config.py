"""Общая конфигурация RAG-пайплайна QuantumForge.

Единый источник правды для эмбеддера, путей и параметров чанкинга — используется
и при сборке индекса (build_index.py), и ботом (Task 4). Так гарантируется
ключевое правило: запрос и документы кодируются ОДНОЙ моделью с одной метрикой.
"""
from __future__ import annotations

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "knowledge_base"
INDEX_DIR = ROOT / "faiss_index"

# --- Эмбеддер ---------------------------------------------------------------
# Qwen3-Embedding-0.6B: multilingual (RU+EN в одном пространстве), 1024-dim,
# контекст 32K, Apache-2.0, instruction-aware.
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
# Пиннинг ревизии модели = воспроизводимость + защита от supply-chain (модель
# может «уехать» под нами). См. конспект «Генерация эмбеддингов» §6.
EMBED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"  # пин commit SHA снапшота

# Qwen3 instruction-aware: к ЗАПРОСУ добавляется инструкция, ДОКУМЕНТЫ кодируются
# как есть. Это аналог префиксов query/passage у e5/bge.
QUERY_PROMPT = (
    "Instruct: Given a user question, retrieve passages from the QuantumForge "
    "knowledge base that answer it.\nQuery: "
)

# --- Чанкинг ----------------------------------------------------------------
# Размер/overlap считаются в ТОКЕНАХ модели (не в символах). Корпус атомарный:
# большинство документов укладывается в один чанк, длинные дробятся с перекрытием.
CHUNK_SIZE = 512      # токенов (≤ лимита эмбеддера 32K — «тихого обреза» нет)
CHUNK_OVERLAP = 64    # ~12.5% длины чанка


def get_embeddings(show_progress: bool = False) -> HuggingFaceEmbeddings:
    """Сконфигурированный эмбеддер Qwen3 (нормировка → косинус)."""
    kwargs = {
        "model_name": EMBED_MODEL,
        "model_kwargs": {"revision": EMBED_REVISION},
        "encode_kwargs": {"normalize_embeddings": True, "batch_size": 16},
        "show_progress": show_progress,
    }
    # Инструкция для запроса — только если версия langchain-huggingface её
    # поддерживает; иначе запрос кодируется без префикса (приемлемый фолбэк).
    if "query_encode_kwargs" in getattr(HuggingFaceEmbeddings, "model_fields", {}):
        kwargs["query_encode_kwargs"] = {
            "normalize_embeddings": True,
            "prompt": QUERY_PROMPT,
        }
    return HuggingFaceEmbeddings(**kwargs)
