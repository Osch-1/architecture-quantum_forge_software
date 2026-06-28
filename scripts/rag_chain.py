"""Ядро RAG-бота: retrieval из FAISS + генерация ответа LLM через Ollama.

Переиспользует конфиг Task 3 (тот же эмбеддер и тот же индекс), поэтому запрос
кодируется так же, как документы. Цепочка собрана на LangChain:
ChatPromptTemplate | ChatOllama | StrOutputParser. Источники (метаданные чанков)
возвращаются вместе с ответом — для цитирования под SOC 2.

Task 5: на онлайн-путь навешаны слои защиты от prompt injection (см. security.py).
Каждый слой включается/выключается через объект Defenses, что и даёт демонстрацию
«без защиты» / «с защитой».

Как библиотека:
    from rag_chain import RagBot
    bot = RagBot()
    res = bot.ask("Кто такой Корвен из Драэля?")
    print(res.answer)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import build_prompt  # noqa: E402
from rag_config import (  # noqa: E402
    INDEX_DIR,
    OLLAMA_MODEL,
    RETRIEVE_K,
    get_embeddings,
    get_llm,
)
from security import (  # noqa: E402
    Defenses,
    guard_output,
    is_malicious,
    sanitize_chunk,
)


@dataclass
class Source:
    """Один процитированный фрагмент (для блока «Источники» в интерфейсе)."""

    n: int
    title: str
    source: str


@dataclass
class Answer:
    question: str
    answer: str       # секция ОТВЕТ — то, что показываем пользователю
    reasoning: str    # секция РАССУЖДЕНИЕ — Chain-of-Thought, можно скрыть
    sources: list[Source]
    raw: str                              # полный ответ модели целиком (для отладки)
    notes: list[str] = field(default_factory=list)  # что сделали слои защиты


def _split_sections(raw: str) -> tuple[str, str]:
    """Делит ответ модели на (рассуждение, ответ) по маркеру 'ОТВЕТ:'."""
    marker = "ОТВЕТ:"
    if marker in raw:
        before, after = raw.split(marker, 1)
        reasoning = before.replace("РАССУЖДЕНИЕ:", "").strip()
        return reasoning, after.strip()
    return "", raw.strip()


def _format_context(pieces: list[tuple[str, dict]]) -> tuple[str, list[Source]]:
    """Собирает пронумерованные фрагменты и список источников из метаданных.

    pieces — список (текст_чанка, метаданные); текст уже мог быть очищен
    sanitize_chunk, поэтому работаем с ним, а не с исходным Document.
    """
    blocks: list[str] = []
    sources: list[Source] = []
    for i, (text, meta) in enumerate(pieces, start=1):
        title = meta.get("title", "без названия")
        src = meta.get("source", "?")
        blocks.append(f"[{i}] {title} (источник: {src})\n{text}")
        sources.append(Source(n=i, title=title, source=src))
    return "\n\n".join(blocks), sources


class RagBot:
    """RAG-бот: один раз загружает индекс и модель, дальше отвечает на вопросы."""

    def __init__(self, k: int = RETRIEVE_K, defenses: Defenses | None = None) -> None:
        self.k = k
        self.model = OLLAMA_MODEL
        self.defenses = defenses or Defenses.all_on()
        embeddings = get_embeddings()
        # allow_dangerous_deserialization: index.pkl — наш собственный артефакт
        # сборки (см. build_index.py), не сторонний файл.
        self.store = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        # LLM создаём один раз; промпт собираем на каждый запрос — он зависит от
        # того, включён ли безопасный pre-prompt.
        self.llm = get_llm()
        self.parser = StrOutputParser()

    def ask(self, question: str, defenses: Defenses | None = None) -> Answer:
        """Отвечает на вопрос с учётом включённых слоёв защиты."""
        d = defenses or self.defenses
        notes: list[str] = []

        docs = self.store.similarity_search(question, k=self.k)

        # Слой: фильтр чанков-троянов (целиком вредоносные выбрасываем).
        if d.filter_chunks:
            kept = []
            for doc in docs:
                if is_malicious(doc.page_content):
                    notes.append(
                        f"фильтр отбросил чанк с инъекцией: {doc.metadata.get('source', '?')}"
                    )
                else:
                    kept.append(doc)
            docs = kept

        # Слой: чистка инструкций из оставшихся чанков. Document не мутируем
        # (это объекты из docstore FAISS) — работаем с копией текста.
        pieces: list[tuple[str, dict]] = []
        for doc in docs:
            text = doc.page_content
            if d.sanitize:
                cleaned = sanitize_chunk(text)
                if cleaned != text:
                    notes.append(
                        f"санитизация очистила чанк: {doc.metadata.get('source', '?')}"
                    )
                    text = cleaned
            pieces.append((text, doc.metadata))

        context, sources = _format_context(pieces)

        chain = build_prompt(secure=d.preprompt) | self.llm | self.parser
        raw = chain.invoke({"context": context, "question": question})
        reasoning, answer = _split_sections(raw)

        # Слой: post-guard — маскируем секрет, если он всё же просочился в ответ
        # или в рассуждение (CoT тоже показывается по :think).
        if d.output_guard:
            g_answer, g_reasoning = guard_output(answer), guard_output(reasoning)
            if g_answer != answer or g_reasoning != reasoning:
                notes.append("post-guard замаскировал секрет в ответе")
            answer, reasoning = g_answer, g_reasoning

        return Answer(question, answer, reasoning, sources, raw, notes)
