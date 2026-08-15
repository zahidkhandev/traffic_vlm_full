FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY README.md /app/README.md
COPY qwen_auto_qc /app/qwen_auto_qc
COPY configs /app/configs
COPY run_cli.py /app/run_cli.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

CMD ["python", "run_cli.py", "run", "--config", "configs/local.yaml"]
