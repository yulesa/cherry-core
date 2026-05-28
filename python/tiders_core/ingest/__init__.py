"""Blockchain data ingestion module.

Provides a streaming interface for ingesting blockchain data from multiple
provider backends (SQD, HyperSync, RPC). Supports both
EVM and SVM chain queries with configurable retry, rate limiting, and
buffering options.
"""

from typing import Dict, Optional
import dataclasses
from dataclasses import dataclass
import pyarrow
from enum import Enum
from . import evm, svm
import tiders_core.tiders_core as cc


class ProviderKind(str, Enum):
    """Supported blockchain data provider backends.

    Attributes:
        SQD: SQD Network (formerly Subsquid) data lake provider.
        HYPERSYNC: Envio HyperSync provider for EVM chains.
        RPC: Direct JSON-RPC node connection.
    """

    SQD = "sqd"
    HYPERSYNC = "hypersync"
    RPC = "rpc"


class QueryKind(str, Enum):
    """Blockchain virtual machine type for queries.

    Attributes:
        EVM: Ethereum Virtual Machine (Ethereum, Polygon, Arbitrum, etc.).
        SVM: Solana Virtual Machine.
    """

    EVM = "evm"
    SVM = "svm"


@dataclass
class Query:
    """A blockchain data query combining the VM kind with query parameters.

    Attributes:
        kind: The blockchain VM type (EVM or SVM).
        params: The query parameters, either an ``evm.Query`` or ``svm.Query`` instance
            matching the specified kind.
    """

    kind: QueryKind
    params: evm.Query | svm.Query


@dataclass
class ProviderConfig:
    """Configuration for a blockchain data provider.

    Attributes:
        kind: The provider backend to use.
        url: The provider endpoint URL. If None, uses the provider's default.
        bearer_token: Authentication token for protected APIs.
        max_num_retries: Maximum number of retries for failed requests.
        retry_backoff_ms: Delay increase between retries in milliseconds.
        retry_base_ms: Base retry delay in milliseconds.
        retry_ceiling_ms: Maximum retry delay in milliseconds.
        req_timeout_millis: Request timeout in milliseconds.
        stop_on_head: If True, stop when reaching the chain head. If False,
            keep polling for new blocks indefinitely.
        head_poll_interval_millis: How frequently (in ms) to poll for new blocks
            when streaming live data.
        buffer_size: Number of responses to buffer before sending to the consumer.
        compute_units_per_second: (RPC only) Rate limit in compute units per second.
        batch_size: (RPC only) Number of blocks fetched per batch.
        reorg_safe_distance: (RPC only) Number of blocks behind head considered
            safe from chain reorganizations.
        trace_method: (RPC only) Trace API method, either ``"trace_block"`` or
            ``"debug_trace_block_by_number"``.
    """

    kind: ProviderKind
    url: Optional[str] = None
    bearer_token: Optional[str] = None
    max_num_retries: Optional[int] = None
    retry_backoff_ms: Optional[int] = None
    retry_base_ms: Optional[int] = None
    retry_ceiling_ms: Optional[int] = None
    req_timeout_millis: Optional[int] = None
    stop_on_head: bool = False
    head_poll_interval_millis: Optional[int] = None
    buffer_size: Optional[int] = None
    compute_units_per_second: Optional[int] = None
    batch_size: Optional[int] = None
    reorg_safe_distance: Optional[int] = None
    trace_method: Optional[str] = None


class ResponseStream:
    """Async iterator over blockchain data responses.

    Each call to ``next()`` yields a dictionary mapping table names (e.g.
    ``"blocks"``, ``"transactions"``, ``"logs"``) to PyArrow RecordBatches
    containing the data for that block range.
    """

    def __init__(self, inner):
        self.inner = inner

    def close(self):
        """Close the stream and release resources."""
        self.inner.close()

    async def next(self) -> Optional[Dict[str, pyarrow.RecordBatch]]:
        """Fetch the next batch of data from the stream.

        Returns:
            A dictionary mapping table names to RecordBatches, or None if the
            stream is exhausted.
        """
        return await self.inner.next()

    @property
    def from_block(self) -> int:
        """The configured start block of the query."""
        return self.inner.from_block

    @property
    def to_block(self) -> Optional[int]:
        """The configured end block of the query, or ``None`` if streaming to head."""
        return self.inner.to_block

    @property
    def last_block(self) -> Optional[int]:
        """Highest block already returned to the caller via ``next()``.

        ``None`` before the first batch is consumed, or while no returned
        batch has carried block-id data.
        """
        return self.inner.last_block


def start_stream(cfg: ProviderConfig, query: Query) -> ResponseStream:
    """Start streaming blockchain data from a provider.

    Creates a connection to the specified provider and begins fetching data
    matching the query parameters. Results are returned as an async
    ``ResponseStream``.

    Args:
        cfg: Provider configuration including endpoint URL and connection settings.
        query: The data query specifying block range, filters, and field selections.

    Returns:
        A ``ResponseStream`` that yields batches of blockchain data as
        ``Dict[str, pyarrow.RecordBatch]``.
    """
    # PyO3 extracts enum fields via Python's str(), which in Python 3.13+ returns
    # "ProviderKind.RPC" instead of "rpc" for StrEnum members. Normalize to .value.
    if isinstance(cfg.kind, Enum):
        cfg = dataclasses.replace(cfg, kind=cfg.kind.value)
    if isinstance(query.kind, Enum):
        query = dataclasses.replace(query, kind=query.kind.value)
    inner = cc.ingest.start_stream(cfg, query)
    return ResponseStream(inner)
