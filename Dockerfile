# Small official Python image, same version as the project (.python-version).
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from its official image, pinned to the local version.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# 1) Dependencies only. This layer is cached and only rebuilt when
#    pyproject.toml or uv.lock change, which makes code-only rebuilds fast.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 2) The project itself: code, README (referenced by pyproject.toml) and model.
COPY README.md ./
COPY src ./src
COPY models ./models
RUN uv sync --locked --no-dev

EXPOSE 8000

# 0.0.0.0 makes the server reachable from outside the container.
# Use the port given by the platform (PORT), or 8000 when running locally.
CMD ["sh", "-c", "/app/.venv/bin/uvicorn credit_risk_api.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
