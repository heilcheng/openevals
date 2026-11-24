FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock models.toml ./
COPY crates ./crates
RUN cargo build --release --locked -p openevals-cli

FROM debian:bookworm-slim
# python3 is needed to execute HumanEval programs; ca-certificates for HTTPS endpoints.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 openevals
COPY --from=build /src/target/release/openevals /usr/local/bin/openevals
COPY configs /app/configs
COPY models.toml /app/models.toml
USER openevals
WORKDIR /app
ENV HF_HOME=/app/cache
VOLUME ["/app/data", "/app/results"]
HEALTHCHECK NONE
ENTRYPOINT ["openevals"]
CMD ["--help"]
LABEL org.opencontainers.image.source="https://github.com/heilcheng/openevals" \
      org.opencontainers.image.description="Benchmarking suite for open-weight language models" \
      org.opencontainers.image.licenses="MIT"
