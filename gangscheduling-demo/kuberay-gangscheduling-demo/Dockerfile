FROM astral/uv:0.8-python3.11-trixie-slim

WORKDIR /opt/app

ENV PYTHONUNBUFFERED="1"
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --locked
ENV PATH="/opt/app/.venv/bin:$PATH"

COPY *.py ./
