FROM astral/uv:0.8-python3.11-trixie-slim

WORKDIR /opt/app

ENV PYTHONUNBUFFERED="1"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked
ENV PATH="/opt/app/.venv/bin:$PATH"

COPY *.py ./
