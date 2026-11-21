"""
Graph manipulation and ingestion layer.
"""

from .subgraph_engine import RealTimeGraphEngine

try:
    from .graph_builder import TransactionGraphBuilder
except (ImportError, ModuleNotFoundError):
    TransactionGraphBuilder = None

try:
    from .neo4j_client import Neo4jClient
except (ImportError, ModuleNotFoundError):
    Neo4jClient = None

__all__ = ["RealTimeGraphEngine", "TransactionGraphBuilder", "Neo4jClient"]

