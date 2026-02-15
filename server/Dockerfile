FROM rust:1-slim-bookworm AS build
WORKDIR /workspace
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY core ./core
COPY cli ./cli
COPY server ./server
RUN cargo build --release -p server

FROM debian:bookworm-slim AS runtime
RUN useradd --system --no-create-home --uid 10001 memoria
COPY --from=build /workspace/target/release/server /usr/local/bin/server
USER memoria
ENTRYPOINT ["/usr/local/bin/server"]
