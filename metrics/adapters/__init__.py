"""Optional adapters for metric implementations maintained elsewhere."""

from .elastic_srvft import (
    DEFAULT_ELASTIC_SRVFT_CHECKOUT,
    ElasticSRVFTDependencyError,
    ElasticSRVFTError,
    ElasticSRVFTNotConfigured,
    ElasticSRVFTResult,
    ElasticSRVFTTreeDiagnostics,
    ElasticSRVFTUnsupportedTree,
    elastic_srvft_distance,
    elastic_srvft_tree_diagnostics,
)

__all__ = [
    "DEFAULT_ELASTIC_SRVFT_CHECKOUT",
    "ElasticSRVFTDependencyError",
    "ElasticSRVFTError",
    "ElasticSRVFTNotConfigured",
    "ElasticSRVFTResult",
    "ElasticSRVFTTreeDiagnostics",
    "ElasticSRVFTUnsupportedTree",
    "elastic_srvft_distance",
    "elastic_srvft_tree_diagnostics",
]
