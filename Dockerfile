FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# Python 3.12 + system deps
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
RUN python -m venv $VIRTUAL_ENV && \
    chown -R ${USERNAME}:${USERNAME} $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

COPY pyproject.toml requirements.txt ./
COPY src/ src/
RUN pip install --no-cache-dir --break-system-packages -e ".[cuda]"

COPY . .
RUN chown -R ${USERNAME}:${USERNAME} /workspace

USER ${USERNAME}

CMD ["bash"]
