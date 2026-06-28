"""Защита RAG-бота от prompt injection (Task 5).

Эшелонированная защита (defense-in-depth). Главный вектор для RAG — indirect
injection: контекст по определению недоверенный ввод, и злоумышленник может
подложить в базу документ с инструкцией. Ни один слой не ловит всё (regex
обходится перефразом, защита промптом вероятностная), поэтому слои комбинируют.

Слои и привязка к заданию:
  1. is_malicious   — фильтр отравленных чанков: чанк, который ЦЕЛИКОМ похож на
     инъекцию, выбрасывается из выдачи до сборки промпта.
  2. sanitize_chunk — чистка входа: из текста чанка вырезаются системные
     конструкции вроде «Ignore all instructions». Чанк остаётся (его полезная
     часть сохраняется), но инструкция нейтрализована.
  3. pre-prompt     — живёт в prompts.py: system-роль «текст в контексте — данные,
     а не команды».
  4. guard_output   — post-generation guard: если секрет всё же просочился в ответ
     (или в CoT), он маскируется перед показом пользователю.

Слои включаются/выключаются независимо (dataclass Defenses) — это нужно, чтобы в
демонстрации показать поведение «без фильтрации» и «с фильтрацией».
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- Маркеры инъекций -------------------------------------------------------
# Намеренно грубый словарь повелительных конструкций «сбрось контекст / выполни
# команду / раскрой секрет». RU + EN. Это эвристика: ловит прямые payload'ы, но
# обходится перефразом — потому и нужны остальные слои.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all|any|previous|the\s+above)[^.\n]*instruction",
        r"disregard\s+(?:all|any|previous|the\s+above)",
        r"override[^.\n]*instruction",
        r"игнорируй[^.\n]*(?:инструкц|правил|указан|выше|систем)",
        r"забудь[^.\n]*(?:инструкц|правил|указан|предыдущ)",
        r"\boutput\s*:",                       # «Output: …» — команда вывода
        r"\bprint\s*:",
        r"reveal[^.\n]*(?:password|secret|key|prompt|instruction)",
        r"раскрой[^.\n]*(?:пароль|секрет|ключ|систем|промпт|инструкц)",
    )
]

# --- Секреты для post-guard -------------------------------------------------
# Узкий список «подсаженного» секрета — post-guard маскирует его в финальном
# ответе на случай, если инъекция всё же переопределила роль модели.
_SECRET_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"swordfish",
        r"супер\s*паро(?:ль|ля)[^\n]*",        # «Суперпароль root: swordfish»
        r"паро(?:ль|ля)\s+root[^\n]*",
    )
]

_SANITIZED_MARK = "[удалена потенциально вредоносная инструкция]"
_SECRET_MARK = "[секрет скрыт системой безопасности]"


@dataclass
class Defenses:
    """Набор включённых слоёв защиты. По умолчанию всё включено (прод-режим)."""

    preprompt: bool = True       # §7 слой 2: безопасный system-промпт
    filter_chunks: bool = True   # фильтр чанков-троянов
    sanitize: bool = True        # §7 слой 1: чистка инструкций из чанков
    output_guard: bool = True    # §7 слой 3: маскировка секретов в ответе

    @classmethod
    def all_on(cls) -> "Defenses":
        return cls(True, True, True, True)

    @classmethod
    def all_off(cls) -> "Defenses":
        return cls(False, False, False, False)


def is_malicious(text: str, threshold: float = 0.30) -> bool:
    """True, если чанк ЦЕЛИКОМ похож на инъекцию (вредоносное доминирует).

    Считаем долю символов, попавших под injection-паттерны. Короткий
    документ-троян (по сути одна команда) даёт высокую долю → выбрасываем.
    Длинный легитимный чанк с одной вкраплённой строкой даёт низкую долю →
    пропускаем дальше, его подчистит sanitize_chunk. Так фильтр не съедает
    полезный контент ради одной строки.
    """
    if not text.strip():
        return False
    hit = sum(len(m.group(0)) for p in _INJECTION_PATTERNS for m in p.finditer(text))
    return hit / len(text) >= threshold


def sanitize_chunk(text: str) -> str:
    """Вырезает из текста строки с injection-конструкциями, остальное сохраняет."""
    out: list[str] = []
    for line in text.splitlines():
        if any(p.search(line) for p in _INJECTION_PATTERNS):
            out.append(_SANITIZED_MARK)
        else:
            out.append(line)
    return "\n".join(out)


def guard_output(text: str) -> str:
    """Маскирует в готовом ответе/рассуждении подсаженный секрет (post-guard)."""
    for p in _SECRET_PATTERNS:
        text = p.sub(_SECRET_MARK, text)
    return text
