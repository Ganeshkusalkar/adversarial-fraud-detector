import threading
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("RealTimeGraphEngine")


class RealTimeGraphEngine:
    """
    High-performance, thread-safe in-memory graph engine for real-time
    k-hop neighborhood subgraph extraction and streaming graph updates.

    Solves the 'Mock Edges' problem by dynamically assembling the true
    topological context of an incoming transaction before GNN scoring.
    """

    def __init__(self, max_history_nodes: int = 5000, neo4j_client=None):
        self.lock = threading.Lock()
        self.max_history_nodes = max_history_nodes
        self.neo4j_client = neo4j_client

        # In-memory entity indices: card -> list of recent node IDs
        self.card_to_nodes: Dict[int, List[str]] = {}
        # Shared attribute indices: attribute -> list of node IDs (carousel detection)
        self.email_to_nodes: Dict[str, List[str]] = {}
        self.addr_to_nodes: Dict[str, List[str]] = {}

        # Node storage: node_id -> { "features": np.ndarray, "card": int, "time": float }
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # Directed adjacency: node_id -> set of neighbor node_ids
        self.adjacency: Dict[str, set] = {}

        self._seed_reference_topology()

    def _seed_reference_topology(self):
        """
        Seeds baseline reference nodes and structured ring topologies to ensure
        immediate inductive graph capability upon service startup.
        """
        logger.info("Seeding reference graph topology for warm-start serving...")
        np.random.seed(42)

        # Seed sample legitimate history
        for i in range(10):
            card = 14200 + i
            node_id = f"REF_TX_{i:04d}"
            # 345 features with realistic low-risk baseline
            feats = np.zeros(345, dtype=np.float32)
            feats[339] = 50.0 + i * 10.0  # TransactionAmt
            feats[340] = 1.0  # C1
            self.nodes[node_id] = {"features": feats, "card": card, "time": time.time() - 3600 + i}
            self.card_to_nodes[card] = [node_id]
            self.adjacency[node_id] = set()

        # Seed a known card-rotation syndicate (cards 19901-19905 sharing the same domain)
        shared_domain = "protonmail.com"
        syndicate_nodes = []
        for i in range(5):
            card = 19901 + i
            node_id = f"RING_TX_{i:04d}"
            feats = np.zeros(345, dtype=np.float32)
            feats[339] = 250.0  # Structured amount
            feats[341] = 5.0    # Elevated C2 velocity
            feats[343] = 4.0    # High node degree
            self.nodes[node_id] = {"features": feats, "card": card, "time": time.time() - 1800 + i}
            self.card_to_nodes[card] = [node_id]
            self.adjacency[node_id] = set()
            syndicate_nodes.append(node_id)

        # Link the syndicate nodes into a ring topology
        for i in range(len(syndicate_nodes)):
            src = syndicate_nodes[i]
            dst = syndicate_nodes[(i + 1) % len(syndicate_nodes)]
            self.adjacency[src].add(dst)
            self.adjacency[dst].add(src)

        self.email_to_nodes[shared_domain] = syndicate_nodes
        logger.info(
            f"Reference graph initialized with {len(self.nodes)} nodes across "
            f"{len(self.card_to_nodes)} unique entities."
        )

    def extract_subgraph(
        self,
        card1: int,
        target_features: np.ndarray,
        p_emaildomain: Optional[str] = None,
        addr1: Optional[str] = None,
        max_hops: int = 2,
        max_neighbors: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Extracts a localized k-hop neighborhood around the target transaction/card.

        Returns:
            x_subgraph: Node feature matrix of shape [N, 345] where index 0 is target.
            edge_index: Graph connectivity tensor [2, E] connecting the subgraph.
            target_idx: Always 0 (the index of the incoming transaction).
        """
        with self.lock:
            # Flatten target_features to 1D (345,)
            target_vec = target_features.flatten().astype(np.float32)

            neighbor_ids = []

            # 1. Look up prior transactions from the same card (temporal history)
            if card1 in self.card_to_nodes:
                neighbor_ids.extend(self.card_to_nodes[card1][-max_neighbors:])

            # 2. Look up cross-card risk edges (shared email domain or billing address)
            if p_emaildomain and p_emaildomain in self.email_to_nodes:
                domain_neighbors = [
                    nid for nid in self.email_to_nodes[p_emaildomain]
                    if nid not in neighbor_ids and self.nodes.get(nid, {}).get("card") != card1
                ]
                neighbor_ids.extend(domain_neighbors[:5])

            if addr1 and str(addr1) in self.addr_to_nodes:
                addr_neighbors = [
                    nid for nid in self.addr_to_nodes[str(addr1)]
                    if nid not in neighbor_ids
                ]
                neighbor_ids.extend(addr_neighbors[:5])

            # Deduplicate and limit neighbor count
            neighbor_ids = list(dict.fromkeys(neighbor_ids))[:max_neighbors]

            # Cold-start inductive case: no prior network context found
            if not neighbor_ids:
                x_subgraph = target_vec.reshape(1, -1)
                # Self-loop edge for single node
                edge_index = np.zeros((2, 1), dtype=np.int64)
                return x_subgraph, edge_index, 0

            # Subgraph Assembly: Node 0 is the incoming target transaction
            subgraph_features = [target_vec]
            node_id_to_idx = {"TARGET": 0}

            for idx, nid in enumerate(neighbor_ids, start=1):
                if nid in self.nodes:
                    subgraph_features.append(self.nodes[nid]["features"])
                    node_id_to_idx[nid] = idx

            x_subgraph = np.vstack(subgraph_features).astype(np.float32)

            # Build edges connecting target to neighbors and neighbors amongst themselves
            edges_src = []
            edges_dst = []

            # Connect target node to its direct card/shared neighbors (bidirectional)
            for nid in neighbor_ids:
                if nid in node_id_to_idx:
                    n_idx = node_id_to_idx[nid]
                    edges_src.extend([0, n_idx])
                    edges_dst.extend([n_idx, 0])

            # Add internal structural edges between neighbors
            for nid in neighbor_ids:
                if nid in self.adjacency and nid in node_id_to_idx:
                    src_idx = node_id_to_idx[nid]
                    for neighbor_nid in self.adjacency[nid]:
                        if neighbor_nid in node_id_to_idx:
                            dst_idx = node_id_to_idx[neighbor_nid]
                            edges_src.append(src_idx)
                            edges_dst.append(dst_idx)

            if len(edges_src) == 0:
                edge_index = np.zeros((2, 1), dtype=np.int64)
            else:
                edge_index = np.array([edges_src, edges_dst], dtype=np.int64)

            return x_subgraph, edge_index, 0

    def add_transaction(
        self,
        tx_id: str,
        card1: int,
        target_features: np.ndarray,
        p_emaildomain: Optional[str] = None,
        addr1: Optional[str] = None,
        timestamp: Optional[float] = None,
    ):
        """
        Asynchronously commits a scored transaction and its structural relationships
        into the live in-memory graph.
        """
        with self.lock:
            ts = timestamp or time.time()
            feats = target_features.flatten().astype(np.float32)

            self.nodes[tx_id] = {"features": feats, "card": card1, "time": ts}
            self.adjacency[tx_id] = set()

            # Connect temporal edge to previous transaction on this card
            if card1 not in self.card_to_nodes:
                self.card_to_nodes[card1] = []
            elif self.card_to_nodes[card1]:
                last_tx = self.card_to_nodes[card1][-1]
                if last_tx in self.nodes:
                    self.adjacency[tx_id].add(last_tx)
                    self.adjacency[last_tx].add(tx_id)

            self.card_to_nodes[card1].append(tx_id)

            # Update shared entity indices
            if p_emaildomain and p_emaildomain.lower() not in ("unknown", "missing", "nan"):
                domain_key = p_emaildomain.lower()
                if domain_key not in self.email_to_nodes:
                    self.email_to_nodes[domain_key] = []
                self.email_to_nodes[domain_key].append(tx_id)

            if addr1:
                addr_key = str(addr1)
                if addr_key not in self.addr_to_nodes:
                    self.addr_to_nodes[addr_key] = []
                self.addr_to_nodes[addr_key].append(tx_id)

            # Evict oldest nodes if capacity is exceeded (Sliding Window)
            if len(self.nodes) > self.max_history_nodes:
                oldest_id = next(iter(self.nodes))
                old_card = self.nodes[oldest_id]["card"]
                del self.nodes[oldest_id]
                self.adjacency.pop(oldest_id, None)
                if old_card in self.card_to_nodes and oldest_id in self.card_to_nodes[old_card]:
                    self.card_to_nodes[old_card].remove(oldest_id)

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_edges = sum(len(adj) for adj in self.adjacency.values()) // 2
            return {
                "total_nodes": len(self.nodes),
                "total_edges": total_edges,
                "tracked_cards": len(self.card_to_nodes),
                "tracked_domains": len(self.email_to_nodes),
            }

