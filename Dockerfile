# syntax=docker/dockerfile:1.7 c

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Le modèle est pré-staged par prepare_build.py — un seul run, rien d'autre
COPY build_model/ /opt/model/

# Deps déclarées par le modèle + serveur Flask
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /opt/model/requirements.txt \
                "flask>=3.0,<4.0" "gunicorn>=23.0,<24.0" "python-dotenv"

COPY app.py /app/app.py
COPY data/dataset.xlsx /app/data/dataset.xlsx

EXPOSE 8080

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${PORT} app:app"]
