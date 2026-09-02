# Build with the repository root as the context. To bundle models in the image,
# place the three required GGUFs in examples/lp_server/models/ first:
# docker build -f examples/lp_server/Dockerfile -t parakeet-lp-server:cpu .
#
# For CUDA, supply matching NVIDIA CUDA build/runtime bases and enable ggml CUDA:
# docker build -f examples/lp_server/Dockerfile -t parakeet-lp-server:cuda \
#   --build-arg BUILD_BASE=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
#   --build-arg RUNTIME_BASE=nvidia/cuda:13.0.1-runtime-ubuntu24.04 \
#   --build-arg "CMAKE_EXTRA_ARGS=-DPARAKEET_GGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON" .

ARG BUILD_BASE=ubuntu:24.04
ARG RUNTIME_BASE=ubuntu:24.04

FROM ${BUILD_BASE} AS build

ARG CMAKE_EXTRA_ARGS=""
ARG CUDA_ARCHS=""

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# The project applies ggml patches during CMake configuration. Ensure the
# vendored submodule is a Git worktree even when the Docker context omitted it.
RUN rm -rf third_party/ggml/.git && git -C third_party/ggml init -q

# Keep CPU images portable. CMAKE_EXTRA_ARGS can enable a GPU backend.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DPARAKEET_BUILD_CLI=OFF \
        -DPARAKEET_BUILD_SERVER=OFF \
        -DPARAKEET_BUILD_LP_SERVER=ON \
        -DPARAKEET_BUILD_TESTS=OFF \
        ${CMAKE_EXTRA_ARGS} \
        ${CUDA_ARCHS:+"-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}"} \
    && cmake --build build --target parakeet-lp-server --parallel "$(nproc)"

RUN mkdir -p /install/bin /install/lib \
    && cp build/examples/lp_server/parakeet-lp-server /install/bin/ \
    && find build -name '*.so*' -exec cp -av {} /install/lib/ \;

FROM ${RUNTIME_BASE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 parakeet \
    && useradd --uid 10001 --gid parakeet --create-home --shell /usr/sbin/nologin parakeet

COPY --from=build /install/lib/ /usr/local/lib/
COPY --from=build /install/bin/parakeet-lp-server /usr/local/bin/
# Model files are intentionally named explicitly so a missing or unexpected
# model fails the image build instead of producing an unusable runtime image.
COPY examples/lp_server/models/stt_ar_fastconformer_hybrid_large_pcd_v1.0-q4_0.gguf /models/
COPY examples/lp_server/models/parakeet-tdt_ctc-110m-q4_0.gguf /models/
COPY examples/lp_server/models/ecapa-lid-voxlingua107.gguf /models/
RUN ldconfig

WORKDIR /models
USER parakeet

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8080/readyz || exit 1

ENTRYPOINT ["parakeet-lp-server", "--host", "0.0.0.0"]
