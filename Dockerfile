# The pipeline's complete environment: Python side from uv.lock, plus the two
# native binaries it shells out to -- `uv sync` alone cannot provide `pymol`
# (no stable wheel on PyPI) or `hhfilter` (not on PyPI at all). The structure
# predictor in proteusfold/ is not in this image; it needs its own environment.
#
# Build:
#   docker build -t proteus .
#
# Run, with your data and the weight caches mounted in:
#   docker run --rm -it --gpus all \
#     --user "$(id -u):$(id -g)" -e HOME=/tmp \
#     -v "$PWD/examples":/work/examples \
#     -v "$HOME/.cache/torch":/tmp/.cache/torch \
#     -v "$HOME/.cache/huggingface":/tmp/.cache/huggingface \
#     -e HF_TOKEN \
#     proteus proteus-doctor
#
# Two flags are not optional in practice.
#   --user  Without it the container runs as root and every file the pipeline
#           writes into a bind mount -- labels, embeddings, mutant PDBs, picks
#           -- is owned by root on the host, inside root-owned directories.
#           You then cannot edit or delete your own results without sudo.
#           `-e HOME=/tmp` goes with it: the mapped user has no entry in the
#           image's /etc/passwd, so HOME defaults to `/`, and the caches below
#           would land in a root-owned `/.cache` that the run cannot write.
#   -v ...cache...  The image ships no model weights. ESM2-3B alone is 5.3 GiB
#           and ESM-MSA-1b another 1.3 GiB, so without these mounts every
#           container start downloads them again.
#
# On a cluster without Docker, convert it:
#   docker save proteus -o proteus.tar
#   apptainer build proteus.sif docker-archive://proteus.tar
# Apptainer bind-mounts $HOME and the working directory by default and maps
# your own user, so `--gpus all` becomes `--nv` and the -v flags above are
# mostly unnecessary. The caches then live under your real $HOME.

# Plain Ubuntu, not an nvidia/cuda image. The torch wheels from
# download.pytorch.org/whl/cu126 bundle their own CUDA runtime (cublas, cudnn,
# nccl and the rest arrive as nvidia-*-cu12 wheels), and the driver's user-mode
# libraries are injected by the container runtime, so a CUDA base image adds a
# second, unused copy of all of it: measured, the cudnn-runtime base cost about
# 8 GB of a 16 GB image for nothing. Ubuntu 22.04 matches the manylinux_2_35
# wheels and the hhsuite/pymol packages below.
FROM ubuntu:22.04

# pymol-open-source is in Ubuntu's universe repo and pulls its own Python;
# we only ever call the `pymol` binary from a subprocess, so that is fine and
# keeps the image smaller than a second conda install would.
#   libgl1/libxrender1/libxext6  pymol needs them even headless (-cq)
#   hhsuite                      provides hhfilter
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        hhsuite \
        libgl1 \
        libxext6 \
        libxrender1 \
        pymol \
    && rm -rf /var/lib/apt/lists/*

# Debian's /usr/bin/pymol is a shell script that runs a bare
# `python3 -m pymol.__init__`, so it uses whichever python comes first on
# PATH. Since the venv below goes on PATH, that would be a python with no
# `pymol` module, and every structure_prep call would fail while
# `which pymol` kept saying it was installed. Shadow it with a wrapper that
# gives Debian's script a PATH where the system python wins, and keep that
# script as the source of truth for PYMOL_DATA and friends.
RUN printf '#!/bin/sh\nexec env PATH=/usr/bin:/bin /usr/bin/pymol "$@"\n' \
      > /usr/local/bin/pymol \
    && chmod +x /usr/local/bin/pymol

# Pinned, not :latest: this is the uv that produced uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /usr/local/bin/uv

# uv unpacks every wheel into the venv and also keeps the downloaded wheel in
# its cache. Both would be baked into the image, so torch would be stored
# twice -- several gigabytes for nothing, since nothing in a built image ever
# resolves dependencies again.
ENV UV_NO_CACHE=1

WORKDIR /work

# Dependencies first, so editing source does not re-resolve or re-download
# five gigabytes of wheels. --frozen means uv.lock is used as written and a
# drifted pyproject.toml is an error rather than a silent re-lock.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --python 3.10

COPY proteus ./proteus
COPY tools ./tools
COPY examples ./examples
RUN uv sync --frozen --python 3.10

# Meta's ESM2/ESM-MSA-1b, installed under the name `fair_esm` so it does not
# collide with EvolutionaryScale's `esm`. See tools/install_fair_esm.py.
RUN uv run --frozen python tools/install_fair_esm.py

# So `proteus-doctor`, `proteus-score`, `pymol` and `hhfilter` are all just
# commands, with no activation step.
ENV PATH="/work/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1

# Build-time check that the image is usable. Only the parts that do not need
# the host: both native binaries are actually executed, and the Python side is
# imported. The weight and GPU checks belong to `proteus-doctor` at run time,
# since the image ships no weights and sees no device here.
RUN pymol -cq -d "print('PYMOL_RUNS')" | grep -q PYMOL_RUNS \
    && hhfilter -h 2>&1 | grep -q HHfilter \
    && python -c "import proteus, fair_esm, esm; from proteus.zeroshot import ProteusZeroShot" \
    && proteus-doctor --help > /dev/null

CMD ["bash"]
