FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv binary directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock ./

# Install third-party dependencies without installing the project itself
RUN uv sync --frozen --no-install-project

# Copy README.md (required by hatchling build backend) and project sources
COPY README.md ./
COPY src/ ./src/
COPY sql/ ./sql/
COPY artifacts/ ./artifacts/
COPY docs/ ./docs/

# Install the semcon package into environment
RUN uv sync --frozen

# Expose Dash default port
EXPOSE 8050

# Default command: launch the Dash dashboard bound to all interfaces
CMD ["uv", "run", "semcon-dash", "--host", "0.0.0.0", "--port", "8050"]