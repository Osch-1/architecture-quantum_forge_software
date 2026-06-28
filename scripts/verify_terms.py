"""Проверка «неузнаваемости» базы знаний.

Сканирует knowledge_base/**/*.md и убеждается, что НИ ОДИН канонический термин
из terms_map.json не просочился в корпус. Это доказывает, что модель не сможет
«угадать» ответы по памяти об исходной вселенной.

Выход: код 1, если найдена хотя бы одна утечка; иначе 0.

Использование:
    python scripts/verify_terms.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "knowledge_base"
TERMS = ROOT / "terms_map.json"


def canon_terms() -> list[str]:
    data = json.loads(TERMS.read_text(encoding="utf-8"))
    out: list[str] = []
    for category in data["mappings"].values():
        for entry in category:
            # пропускаем общеупотребимые слова (cat, знак, swallow…): они не
            # «опознают» исходную вселенную и дают ложные срабатывания
            if entry.get("common"):
                continue
            for key in ("canon_en", "canon_ru"):
                if entry.get(key):
                    out.append(entry[key])
    return out


def main() -> None:
    terms = canon_terms()
    docs = sorted(KB.rglob("*.md"))
    leaks: list[tuple[str, str]] = []

    for md in docs:
        text = md.read_text(encoding="utf-8")
        for term in terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE | re.UNICODE):
                leaks.append((md.relative_to(ROOT).as_posix(), term))

    if leaks:
        print(f"LEAK: найдено {len(leaks)} канонических терминов в базе знаний:")
        for path, term in leaks:
            print(f"  {path}: {term!r}")
        sys.exit(1)

    print(f"OK: канонических терминов не найдено в {len(docs)} документах базы знаний.")


if __name__ == "__main__":
    main()
