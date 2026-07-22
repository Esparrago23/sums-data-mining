FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# --extra-index-url (no --index-url) en la MISMA resolucion que requirements.txt:
# con dos `pip install` separados, el segundo puede re-resolver e instalar el
# torch con CUDA del indice default al ver sentence-transformers, deshaciendo
# el CPU-only del primero. Con extra-index-url, pip ve ambos indices a la vez
# y prefiere el wheel CPU-only (mas chico) sin necesitar CUDA/NVIDIA (varios
# GB inutiles en un contenedor sin GPU passthrough).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
RUN python -m spacy download es_core_news_sm

COPY . .
RUN chmod +x /app/docker-entrypoint.sh

CMD ["/app/docker-entrypoint.sh"]
