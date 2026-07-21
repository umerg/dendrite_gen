"""Optional adapters for metric implementations maintained elsewhere."""

from .elastic_srvft import (
    DEFAULT_ELASTIC_SRVFT_CHECKOUT,
    ElasticSRVFTDependencyError,
    ElasticSRVFTError,
    ElasticSRVFTNotConfigured,
    ElasticSRVFTResult,
    ElasticSRVFTUnsupportedTree,
    elastic_srvft_distance,
)

__all__ = [
    "DEFAULT_ELASTIC_SRVFT_CHECKOUT",
    "ElasticSRVFTDependencyError",
    "ElasticSRVFTError",
    "ElasticSRVFTNotConfigured",
    "ElasticSRVFTResult",
    "ElasticSRVFTUnsupportedTree",
    "elastic_srvft_distance",
]
