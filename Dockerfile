FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    libpango-1.0-0 libpangoft2-1.0-0 libgdk-pixbuf-2.0-0 \
    libffi-dev shared-mime-info \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Heavy deps (rarely change — cached layer, ~60min saved on rebuild)
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Light deps (frequently change — fast rebuild)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY alembic.ini .
COPY alembic/ ./alembic/
COPY templates/ ./templates/
COPY tests/ ./tests/
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
