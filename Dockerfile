# vector-toolbox-mcp
#
#   HTTP (default):  docker run -p 127.0.0.1:8000:8000 -e VTB_AUTH_TOKEN=... ghcr.io/salindahulangamuwa/vector-toolbox-mcp
#   stdio:           docker run -i --rm -e VTB_TRANSPORT=stdio ghcr.io/salindahulangamuwa/vector-toolbox-mcp
#
# EXTRAS picks optional dependencies. The default skips `local`
# (sentence-transformers + torch, several GB); add it if you need it:
#   docker build --build-arg EXTRAS=chroma,openai,cohere,local .

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- build ----
FROM python:${PYTHON_VERSION}-slim AS build
ARG EXTRAS=chroma,openai,cohere
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv venv /opt/venv \
 && VIRTUAL_ENV=/opt/venv uv pip install --no-cache ".[${EXTRAS}]"

# -------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG VERSION=dev
LABEL org.opencontainers.image.title="vector-toolbox-mcp" \
      org.opencontainers.image.description="MCP server for vector databases (Pinecone, Chroma)" \
      org.opencontainers.image.source="https://github.com/SalindaHulangamuwa/VectorToolBoxMCP" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"

# Non-root user whose home is the data volume, so Chroma's persistent store
# and its embedding-model cache (~/.cache/chroma) survive container restarts.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /data --no-create-home app \
 && mkdir -p /data && chown app:app /data

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/data \
    VTB_TRANSPORT=http \
    VTB_HOST=0.0.0.0 \
    VTB_PORT=8000 \
    VTB_CHROMA_CLIENT=persistent \
    VTB_CHROMA_PATH=/data/chroma

USER app
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request; sys.exit(0) if os.environ.get('VTB_TRANSPORT','http')!='http' else urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('VTB_PORT','8000'), timeout=4)"]

ENTRYPOINT ["vector-toolbox-mcp"]
