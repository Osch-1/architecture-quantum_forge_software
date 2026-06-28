# QuantumForge RAG

RAG-бот по корпоративной базе знаний вымышленной компании **QuantumForge Software**
(учебный проект, спринт 7). Подробное описание решения и обоснование выбора стека
лежит в [Project_template.md](Project_template.md).

> **Стек:** Python 3.11+, LangChain, **FAISS** (векторный индекс),
> **Sentence-Transformers `Qwen/Qwen3-Embedding-0.6B`** (эмбеддинги, локально),
> **LLM локально через Ollama** (Задание 4). Индекс собирается офлайн, документы
> никуда не уходят — этого требуют GDPR и SOC 2 в условиях кейса.

## Структура репозитория

```text
data/
  knowledge_base/        база знаний: 38 .md-документов (19 EN + 19 RU)
    en/  ru/
  terms_map.json         словарь замен из канона в лексику вымышленного мира
scripts/
  rag_config.py          единый конфиг: эмбеддер, пути, параметры чанкинга
  build_index.py         сборка FAISS-индекса из data/knowledge_base
  replace_terms.py       замена канонических терминов на вымышленные (по terms_map.json)
  verify_terms.py        проверка, что в базе не осталось канонических терминов
faiss_index/             векторный индекс; в git не хранится, собирается заново
pyproject.toml           зависимости проекта
Project_template.md      основной отчёт по заданиям 1–5
```

## Установка

Нужен Python **3.11+**. Из корня репозитория:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install .                          # ставит зависимости из pyproject.toml
```

## Скачать модель эмбеддингов

Веса весят примерно 1.2 ГБ. Скачиваются один раз в кеш Hugging Face, дальше
берутся уже оттуда:

```powershell
hf download Qwen/Qwen3-Embedding-0.6B
```

> Если основной хост Hugging Face недоступен, можно взять зеркало и отключить
> бэкенд Xet:
> `$env:HF_ENDPOINT="https://hf-mirror.com"; $env:HF_HUB_DISABLE_XET="1"` перед командой.

Версия модели зафиксирована на конкретный commit в `scripts/rag_config.py`
(`EMBED_REVISION`) — это нужно для воспроизводимости и чтобы веса не подменили
незаметно.

## Сборка векторного индекса

```powershell
python scripts/build_index.py
```

Скрипт читает `data/knowledge_base/**/*.md`, режет тело документов на чанки с
перекрытием (длину считает в токенах модели), кодирует их эмбеддером Qwen3
(с нормировкой, чтобы получить косинусную близость) и сохраняет FAISS-индекс
в `faiss_index/`.

### Результат сборки (Задание 3)

| Параметр | Значение |
|---|---|
| Модель эмбеддингов | `Qwen/Qwen3-Embedding-0.6B` (1024-dim, контекст 32K, multilingual, Apache-2.0) |
| База знаний | 38 документов (19 EN + 19 RU) |
| Чанкинг | RecursiveCharacterTextSplitter, **512 токенов**, overlap **64** (~12.5%) |
| Чанков (векторов) | **39** |
| Тип индекса FAISS | `IndexFlatIP`, точный поиск; inner product по нормированным векторам = косинус |
| Размер на диске | ~0.2 МБ (`index.faiss` + `index.pkl` с метаданными для цитат) |
| Время сборки | ~62 c (CPU, веса из кеша) |

Рядом с векторами индекс держит метаданные каждого документа
(`source`, `title`, `lang`, `category`, `doc_version` и т.д.). Из них бот собирает
ссылку на источник в ответе, как того требует SOC 2-аудит.

## Проверки базы знаний

```powershell
# перевод канонического текста в лексику вымышленного мира
echo "Geralt cast Igni" | python scripts/replace_terms.py --lang en

# проверка, что канонических терминов в базе не осталось (exit 1, если что-то нашлось)
python scripts/verify_terms.py
```
