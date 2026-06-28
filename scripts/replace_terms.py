"""Применить вымышленный словарь (terms_map.json) к каноническому тексту.

База знаний написана сразу в вымышленной лексике, поэтому этот скрипт —
переиспользуемый инструмент подмены: он переводит ЛЮБОЙ канонический текст
(canon -> fictional) в наш переименованный мир для нужного языка.

Логика: самые длинные термины заменяются первыми (чтобы многословные термины
выигрывали у своих частей); регистр первой буквы сохраняется; границы слова —
по Unicode, чтобы корректно работать с кириллицей.

Использование:
    python scripts/replace_terms.py --lang ru --in source.txt --out out.txt
    echo "Geralt used Igni" | python scripts/replace_terms.py --lang en
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERMS = ROOT / "terms_map.json"


def load_pairs(lang: str) -> list[tuple[str, str]]:
    data = json.loads(TERMS.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for category in data["mappings"].values():
        for entry in category:
            canon, fic = entry.get(f"canon_{lang}"), entry.get(f"fic_{lang}")
            if canon and fic:
                pairs.append((canon, fic))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def _preserve_case(matched: str, repl: str) -> str:
    return repl[:1].upper() + repl[1:] if matched[:1].isupper() else repl


def replace(text: str, pairs: list[tuple[str, str]]) -> str:
    for canon, fic in pairs:
        pattern = re.compile(rf"(?<!\w){re.escape(canon)}(?!\w)", re.IGNORECASE | re.UNICODE)
        text = pattern.sub(lambda m: _preserve_case(m.group(0), fic), text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Canon -> fictional term replacement")
    ap.add_argument("--lang", choices=["ru", "en"], required=True)
    ap.add_argument("--in", dest="inp", help="input file (default: stdin)")
    ap.add_argument("--out", dest="out", help="output file (default: stdout)")
    args = ap.parse_args()

    pairs = load_pairs(args.lang)
    text = Path(args.inp).read_text(encoding="utf-8") if args.inp else sys.stdin.read()
    result = replace(text, pairs)

    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"written: {args.out} ({len(pairs)} term rules, lang={args.lang})", file=sys.stderr)
    else:
        sys.stdout.write(result)


if __name__ == "__main__":
    main()
