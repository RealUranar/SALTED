# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.10-slim-bookworm
ARG OPENMPI_VERSION=4.1.8

# false -> use Open MPI's bundled PMIx
# true  -> build and use EXTERNAL_PMIX_VERSION explicitly
ARG USE_EXTERNAL_PMIX=false
ARG EXTERNAL_PMIX_VERSION=6.1.0

# -----------------------------------------------------------------------------
# Build MPI
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS native-builder

ARG OPENMPI_VERSION
ARG USE_EXTERNAL_PMIX
ARG EXTERNAL_PMIX_VERSION

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        dpkg-dev \
        pkg-config \
        libevent-dev \
        libhwloc-dev \
        libmunge-dev \
        libpmi2-0-dev \
        zlib1g-dev

WORKDIR /tmp/build

# -----------------------------------------------------------------------------
# PMIx -- either build external PMIx or use Open MPI's bundled PMIx
# -----------------------------------------------------------------------------
RUN set -eux; \
    mkdir -p /opt/pmix; \
    if [ "${USE_EXTERNAL_PMIX}" = "true" ]; then \
        MULTIARCH="$(dpkg-architecture -qDEB_HOST_MULTIARCH)"; \
        LIBDIR="/usr/lib/${MULTIARCH}"; \
        echo "Building external PMIx ${EXTERNAL_PMIX_VERSION}"; \
        curl -fsSL \
            "https://github.com/openpmix/openpmix/releases/download/v${EXTERNAL_PMIX_VERSION}/pmix-${EXTERNAL_PMIX_VERSION}.tar.gz" \
            | tar -xz; \
        cd "pmix-${EXTERNAL_PMIX_VERSION}"; \
        ./configure \
            --prefix=/opt/pmix \
            --with-hwloc=/usr \
            --with-hwloc-libdir="${LIBDIR}" \
            --with-libevent=/usr \
            --with-libevent-libdir="${LIBDIR}" \
            --with-munge=/usr \
            --with-munge-libdir="${LIBDIR}"; \
        make -j"$(nproc)"; \
        make install; \
        cd /tmp/build; \
        rm -rf "pmix-${EXTERNAL_PMIX_VERSION}"; \
    else \
        echo "External PMIx disabled; Open MPI will use internal PMIx"; \
    fi

# Open MPI
RUN set -eux; \
    curl -fsSL \
        "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OPENMPI_VERSION}.tar.gz" \
        | tar -xz; \
    cd "openmpi-${OPENMPI_VERSION}"; \
    if [ "${USE_EXTERNAL_PMIX}" = "true" ]; then \
        PMIX_CONFIG="--with-pmix=/opt/pmix"; \
    else \
        PMIX_CONFIG="--with-pmix=internal"; \
    fi; \
    echo "Open MPI PMIx configuration: ${PMIX_CONFIG}"; \
    ./configure \
        --prefix=/opt/mpi \
        --enable-shared \
        --disable-static \
        --disable-debug \
        --enable-builtin-atomics \
        --disable-mpi-fortran \
        --disable-oshmem \
        --with-slurm \
        ${PMIX_CONFIG} \
        --with-hwloc=/usr \
        --with-libevent=/usr \
        --with-zlib=/usr \
        --without-psm \
        --without-psm2; \
    make -j"$(nproc)"; \
    make install-strip; \
    cd /tmp/build; \
    rm -rf "openmpi-${OPENMPI_VERSION}"

ENV PATH=/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib

# Create a runtime-only copy without headers, pkg-config files, static
# archives or documentation.
RUN mkdir -p /opt/runtime \
    && cp -a /opt/mpi /opt/runtime/mpi \
    && cp -a /opt/pmix /opt/runtime/pmix \
    && rm -rf \
        /opt/runtime/mpi/include \
        /opt/runtime/mpi/share/man \
        /opt/runtime/mpi/share/doc \
        /opt/runtime/mpi/lib/pkgconfig \
        /opt/runtime/pmix/include \
        /opt/runtime/pmix/share/man \
        /opt/runtime/pmix/share/doc \
        /opt/runtime/pmix/lib/pkgconfig \
    && find /opt/runtime -type f \
        \( -name '*.a' -o -name '*.la' \) \
        -delete


# -----------------------------------------------------------------------------
# Build the Python environment
# -----------------------------------------------------------------------------
ENV PATH=/opt/venv/bin:/opt/mpi/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/venv

RUN --mount=type=cache,target=/root/.cache/pip \
    MPICC=/opt/mpi/bin/mpicc \
    python -m pip install --no-binary=mpi4py mpi4py

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install \
        --prefer-binary \
        pyscf

WORKDIR /src/SALTED
COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install . \
    && find /opt/venv -type d -name '__pycache__' \
        -prune -exec rm -rf '{}' +

# -----------------------------------------------------------------------------
# Minimal runtime
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libevent-2.1-7 \
        libevent-pthreads-2.1-7 \
        libhwloc15 \
        libmunge2 \
        libpmi2-0 \
        openssh-client \
        zlib1g

COPY --from=native-builder /opt/runtime/mpi /opt/mpi
COPY --from=native-builder /opt/runtime/pmix /opt/pmix
COPY --from=native-builder /opt/venv /opt/venv

ENV PATH=/opt/venv/bin:/opt/mpi/bin:/opt/pmix/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/mpi/lib:/opt/pmix/lib:/opt/pmix/lib64
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /work

CMD ["/bin/bash"]
