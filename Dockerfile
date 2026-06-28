# QuantumForge RAG-бот: Streamlit + FAISS (в процессе) + клиент к Ollama.
FROM python:3.11-slim

# libgomp1 нужен faiss-cpu и torch (OpenMP).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости отдельным слоем (кэш). torch берём CPU-сборкой через доп. индекс,
# чтобы не тянуть CUDA: версия `2.12.1+cpu` удовлетворяет пину из pyproject.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu .

# Код и база знаний.
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY app.py ./

ENV HF_HOME=/app/.hf-cache \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=qwen2.5:7b-instruct

EXPOSE 8501

# При первом запуске собираем индекс (safety_in вычищает подсаженные пароли, в
# индекс они не попадают), дальше берём готовый из тома. Затем поднимаем Streamlit.
CMD ["sh", "-c", "[ -f faiss_index/index.faiss ] || python scripts/build_index.py; exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
