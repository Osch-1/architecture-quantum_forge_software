"""Терминальный интерфейс RAG-бота (REPL).

Запуск (Ollama должна быть запущена, модель скачана):
    python scripts/cli.py

Команды внутри:
    :think  — показывать/скрывать ход рассуждений (CoT); по умолчанию скрыт
    :safe   — включить все слои защиты от prompt injection (по умолчанию)
    :unsafe — выключить защиту (демонстрация утечки «без фильтрации»)
    :exit   — выход (или Ctrl-C / Ctrl-D)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rag_chain import RagBot  # noqa: E402
from security import Defenses  # noqa: E402


def main() -> None:
    print("Загрузка модели и индекса…")
    bot = RagBot()
    show_reasoning = False
    defenses = Defenses.all_on()
    print(
        f"Готово. Модель: {bot.model}. Защита: вкл. "
        "Команды: ':think', ':safe', ':unsafe', ':exit'."
    )

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question == ":exit":
            break
        if question == ":think":
            show_reasoning = not show_reasoning
            print(f"[ход рассуждений: {'показан' if show_reasoning else 'скрыт'}]")
            continue
        if question in (":safe", ":unsafe"):
            defenses = Defenses.all_on() if question == ":safe" else Defenses.all_off()
            print(f"[защита: {'включена' if question == ':safe' else 'ВЫКЛЮЧЕНА'}]")
            continue

        res = bot.ask(question, defenses=defenses)

        for note in res.notes:
            print(f"[защита] {note}")

        if show_reasoning and res.reasoning:
            print("\n--- ход рассуждений (CoT) ---")
            print(res.reasoning)
            print("--- ответ ---")
        print(f"\n{res.answer}")

        if res.sources:
            print("\nИсточники:")
            for s in res.sources:
                print(f"  [{s.n}] {s.title} — {s.source}")


if __name__ == "__main__":
    main()
