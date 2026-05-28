//! Python bindings for the ingest module.
//!
//! Exposes [`start_stream`] which creates a [`ResponseStream`] that yields
//! `dict[str, pyarrow.RecordBatch]` from an async blockchain data stream.

use std::collections::BTreeMap;
use std::pin::Pin;

use anyhow::{Context, Result};
use arrow::pyarrow::ToPyArrow;
use baselib::ingest::{ProviderConfig, Query, StreamItem};
use futures_lite::{Stream, StreamExt};
use pyo3::prelude::*;

/// Registers the `ingest` submodule with `start_stream`.
pub fn ingest_module(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "ingest")?;

    submodule.add_function(wrap_pyfunction!(start_stream, m)?)?;

    m.add_submodule(&submodule)?;

    Ok(())
}

/// Async iterator over blockchain data batches, yielding `dict[str, RecordBatch]` per chunk.
#[pyclass]
#[expect(clippy::type_complexity)]
struct ResponseStream {
    inner: Option<Pin<Box<dyn Stream<Item = Result<StreamItem>> + Send + Sync>>>,
    from_block: u64,
    to_block: Option<u64>,
    last_block: Option<u64>,
}

#[pymethods]
impl ResponseStream {
    /// Configured start block of the query.
    #[getter]
    #[expect(
        clippy::wrong_self_convention,
        reason = "pyo3 getter name must match field"
    )]
    fn from_block(&self) -> u64 {
        self.from_block
    }

    /// Configured end block of the query (`None` if streaming to head).
    #[getter]
    fn to_block(&self) -> Option<u64> {
        self.to_block
    }

    /// Highest block already returned to the caller via [`next`], or `None`
    /// before the first batch (or if no returned batch has carried block data).
    #[getter]
    fn last_block(&self) -> Option<u64> {
        self.last_block
    }

    /// Closes the stream, releasing the underlying provider connection.
    pub fn close(&mut self) {
        self.inner.take();
    }

    /// Returns the next batch of data, or `None` when the stream is exhausted.
    pub async fn next(&mut self) -> PyResult<Option<BTreeMap<String, PyObject>>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(None);
        };

        let item: StreamItem = if let Some(n) = inner.next().await {
            n.context("get next item from inner stream")?
        } else {
            self.inner = None;
            return Ok(None);
        };

        if let Some(lb) = item.last_block {
            self.last_block = Some(self.last_block.map_or(lb, |cur| cur.max(lb)));
        }

        let mut out = BTreeMap::new();

        for (table_name, batch) in item.data {
            let batch =
                Python::with_gil(|py| batch.to_pyarrow(py).context("map result to pyarrow"))?;

            out.insert(table_name, batch);
        }

        Ok(Some(out))
    }
}

#[pyfunction]
fn start_stream(
    provider_config: &Bound<'_, PyAny>,
    query: &Bound<'_, PyAny>,
) -> PyResult<ResponseStream> {
    let cfg: ProviderConfig = provider_config.extract().context("parse provider config")?;
    let query: Query = query.extract().context("parse query")?;

    let (from_block, to_block) = match &query {
        Query::Evm(q) => (q.from_block, q.to_block),
        Query::Svm(q) => (q.from_block, q.to_block),
    };

    let inner = crate::TOKIO_RUNTIME.block_on(async move {
        baselib::ingest::start_stream(cfg, query)
            .await
            .context("start stream")
    })?;

    Ok(ResponseStream {
        inner: Some(inner),
        from_block,
        to_block,
        last_block: None,
    })
}
