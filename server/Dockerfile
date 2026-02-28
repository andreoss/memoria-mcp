FROM rust:1-slim-trixie AS build-fastembed
RUN apt-get update && apt-get install -y --no-install-recommends g++ ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY core ./core
COPY cli ./cli
COPY server ./server
RUN cargo build --release -p server --features fastembed
RUN MEMORIA_TEST_FASTEMBED_CACHE_DIR=/opt/fastembed-cache \
    cargo test -p memoria-core --features fastembed --release \
    real_fastembed_model_produces_a_real_semantic_embedding -- --ignored --nocapture

FROM debian:trixie-slim AS runtime-fastembed
RUN useradd --system --no-create-home --uid 10001 memoria \
    && mkdir -p /home/memoria \
    && chown memoria:memoria /home/memoria
COPY --from=build-fastembed /workspace/target/release/server /usr/local/bin/server
COPY --from=build-fastembed --chown=memoria:memoria /opt/fastembed-cache /opt/fastembed-cache
ENV MEMORIA_FASTEMBED_CACHE_DIR=/opt/fastembed-cache
USER memoria
ENTRYPOINT ["/usr/local/bin/server"]

FROM rust:1-slim-bookworm AS build
WORKDIR /workspace
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY core ./core
COPY cli ./cli
COPY server ./server
RUN cargo build --release -p server

FROM debian:bookworm-slim AS runtime
RUN useradd --system --no-create-home --uid 10001 memoria \
    && mkdir -p /home/memoria \
    && chown memoria:memoria /home/memoria
COPY --from=build /workspace/target/release/server /usr/local/bin/server
USER memoria
ENTRYPOINT ["/usr/local/bin/server"]
