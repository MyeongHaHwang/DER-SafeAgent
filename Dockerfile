# DER-SafeAgent — CPU verification image.
#
# Covers: unit/safety tests, smoke test, artifact checksum verification, and
# regeneration of the manuscript tables/figures from the canonical results.
# It deliberately does NOT include the GPU stack; full model-in-the-loop
# re-execution requires an NVIDIA GPU host (see docs/EXPERIMENTS.md) and is
# run outside this image or in a CUDA base image with `make setup-llm`.
#
# Build:  docker build -t der-safeagent .
# Verify: docker run --rm der-safeagent
# Shell:  docker run --rm -it der-safeagent bash

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: fast verification ladder (tests -> smoke -> checksums -> paper artifacts)
CMD ["bash", "-lc", "make test && make smoke && make verify && make reproduce-paper"]
