# Optional alternative to `python run.py` (docs/SPEC.md §3.1).
#   docker compose up        -> open http://127.0.0.1:8765/
# Only the package, sample run and build files are copied in (never .env, sessions/,
# runs/ or .venv/): secrets reach the container through env_file or the GUI settings,
# which are stored in the socmint-home volume.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SOCMINT_HOME=/home/socmint/.socmint \
    SOCMINT_RUNS_DIR=/app/runs \
    SOCMINT_BIND=0.0.0.0

WORKDIR /app

# CPU-only torch wheels keep the image far smaller than the default CUDA build.
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY run.py ./
COPY socmint ./socmint
COPY data ./data
RUN pip install --no-deps -e . \
 && useradd --create-home --uid 1000 socmint \
 && mkdir -p /app/runs /home/socmint/.socmint \
 && chown -R socmint:socmint /app/runs /home/socmint

USER socmint
EXPOSE 8765
# SOCMINT_BIND=0.0.0.0 is required inside the container so the published port reaches
# the server; docker-compose.yml publishes it on the host's 127.0.0.1 only.
CMD ["python", "-m", "socmint.cli", "gui", "--no-browser", "--port", "8765"]
