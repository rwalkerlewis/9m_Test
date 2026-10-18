# Development image for acoustic-sim (drone pipeline + harbor model).
#
# The repository is bind-mounted at /workspace by docker-compose, so this image holds only the
# toolchain and the Python dependencies; source edits never trigger a rebuild.  The dependency
# layer is keyed on pyproject.toml alone.
#
#   BASE_IMAGE  plain Ubuntu by default.  CuPy gets its CUDA toolkit from pip (the `[ctk]` extra
#               below), and the NVIDIA container runtime supplies only the host driver, so the same
#               image works with and without a GPU.
#   EXTRAS      pip extras to install, e.g. "cuda,dev" (default) or "cuda,dev,ml" for PyTorch.
#   USER_UID/GID  match your host user so files written to the bind mount are yours
#               (dev.sh sets these; VS Code Dev Containers adjusts them automatically).
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-venv python3-pip \
        build-essential \
        libsndfile1 \
        ffmpeg \
        git \
        openssh-client \
        sudo \
        libopenmpi-dev \
        openmpi-bin \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && git config --system --add safe.directory /workspace \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
RUN if getent group ${USER_GID} > /dev/null; then \
        groupmod -n ${USERNAME} $(getent group ${USER_GID} | cut -d: -f1); \
    else \
        groupadd --gid ${USER_GID} ${USERNAME}; \
    fi && \
    if id -u ${USER_UID} > /dev/null 2>&1; then \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m $(id -nu ${USER_UID}); \
    else \
        useradd --no-log-init --uid ${USER_UID} --gid ${USER_GID} -m -s /bin/bash ${USERNAME}; \
    fi && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV && chown -R ${USERNAME}:${USERNAME} $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace
RUN chown ${USERNAME}:${USERNAME} /workspace

# Install the dependencies against an empty stub package so this layer only depends on
# pyproject.toml.  The editable install points at /workspace/src, which is where the real
# source appears once the repository is mounted.
ARG EXTRAS=cuda,dev
COPY pyproject.toml ./
RUN mkdir -p src/acoustic_sim && touch src/acoustic_sim/__init__.py && \
    pip install --no-cache-dir -e ".[${EXTRAS}]" && \
    case ",${EXTRAS}," in \
        *,cuda,*) pip install --no-cache-dir "cupy-cuda12x[ctk]" ;; \
    esac && \
    chown -R ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}

CMD ["bash"]
