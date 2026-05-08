# tiders-core

[![Documentation](https://img.shields.io/badge/documentation-blue?style=for-the-badge&logo=readthedocs)](https://yulesa.github.io/tiders-docs/)
[![PyPI](https://img.shields.io/badge/PyPI-lightgreen?style=for-the-badge&logo=pypi&labelColor=white)](https://pypi.org/project/tiders-core//)
[![Crates.io](https://img.shields.io/badge/crates.io-orange?style=for-the-badge&logo=rust)](https://crates.io/crates/tiders-core)
[![tiders](https://img.shields.io/badge/github-black?style=for-the-badge&logo=github)](https://github.com/yulesa/tiders)
[![tiders-rpc-client](https://img.shields.io/badge/github-black?style=for-the-badge&logo=github)](https://github.com/yulesa/tiders-rpc-client)
[![telegram](https://img.shields.io/badge/telegram-blue?style=for-the-badge&logo=telegram)](https://t.me/tidersindexer)


Core libraries for the [`Tiders`](https://github.com/yulesa/tiders) blockchain data pipeline framework.

`tiders-core` is a Rust workspace that provides the high-performance engine behind Tiders (Python SDK). It handles data ingestion, ABI decoding, type casting, encoding, and schema definitions. Tiders calls into these libraries via [PyO3](https://pyo3.rs/) bindings.

## Crates

| Crate | Purpose |
|---|---|
| `tiders-ingest` | Data provider orchestration and streaming |
| `tiders-evm-decode` | EVM event and function ABI decoding |
| `tiders-svm-decode` | Solana instruction and log decoding |
| `tiders-cast` | Arrow column type casting (blockchain-aware) |
| `tiders-evm-schema` | Arrow schema definitions for EVM data |
| `tiders-svm-schema` | Arrow schema definitions for SVM data |
| `tiders-query` | Query execution and filtering |
| `tiders-core` | Re-export crate aggregating all of the above |
| `tiders-core-python` | PyO3 Python bindings |

## Dependency Graph

```text
tiders-core (re-exports)
├── tiders-ingest
│   ├── tiders-evm-schema
│   ├── tiders-svm-schema
│   └── tiders-rpc-client (optional, for RPC provider)
├── tiders-evm-decode
│   └── tiders-cast
├── tiders-svm-decode
├── tiders-cast
└── tiders-query
```

## Contributing

To develop locally across all repos, clone all three projects side by side:

```bash
git clone https://github.com/yulesa/tiders.git
git clone https://github.com/yulesa/tiders-core.git
git clone https://github.com/yulesa/tiders-rpc-client.git
```

### Building from source

Build `tiders-core`:

```bash
cd tiders-core
cargo build
```

If you're also modifying `tiders-rpc-client` locally, override the crates.io version:

```bash
cargo build --config 'patch.crates-io.tiders-rpc-client.path="../tiders-rpc-client/rust"'
```

Build the Python bindings:

```bash
cd tiders-core/python
maturin develop --uv
# With local tiders-rpc-client override
maturin develop --uv --config 'patch.crates-io.tiders-rpc-client.path="../../tiders-rpc-client/rust"'
```

### Persistent local development

To avoid passing `--config` on every build, add to `tiders-core/Cargo.toml`:

```toml
[patch.crates-io]
tiders-rpc-client = { path = "../tiders-rpc-client/rust" }
```

To use your local `tiders-core` in `tiders`, add to `tiders/pyproject.toml`:

```toml
[tool.uv.sources]
tiders-core = { path = "../tiders-core/python", editable = true }
```

Then sync:

```bash
cd tiders
uv sync
```

## Acknowledgements

Tiders is a fork of [Cherry](https://github.com/steelcake/cherry) and [cherry-core](https://github.com/steelcake/cherry-core), a blockchain data pipeline framework built by the [SteelCake](https://github.com/steelcake) team. Cherry laid the architectural foundation that Tiders builds upon, and we're grateful for their work and the open-source spirit that made this continuation possible.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
