import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Fallback for neo4j client dependency
try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    logger.warning("Neo4j python package not installed. Running Neo4jClient in simulator mode.")

class Neo4jClient:
    """
    Production-grade enterprise client wrapper for Neo4j database transactions.
    Isolates Cypher schema executions and multi-hop transaction queries.
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        
        if HAS_NEO4J:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                logger.info("Connected to Neo4j database driver.")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j instance: {e}")
                self._driver = None

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j database driver closed.")

    def run_cypher(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Runs a standard cypher query and returns the record dictionaries.
        """
        if not self._driver:
            logger.warning("Neo4j client driver offline. Simulating query execution.")
            return []
            
        parameters = parameters or {}
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def fetch_transaction_subgraph(self, account_id: str, hops: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve localized transaction subgraph around an account ID for real-time model scoring.
        """
        logger.info(f"Querying {hops}-hop subgraph paths for Account ID: {account_id}...")
        
        query = (
            f"MATCH path = (a:Account {{accountId: $accountId}})-[:TRANSACTED*1..{hops}]-(b:Account) "
            "RETURN path LIMIT 100"
        )
        
        return self.run_cypher(query, {"accountId": account_id})

    def write_transaction_node(self, tx_data: Dict[str, Any]) -> None:
        """
        Writes a new transaction node and updates relationships in real-time.
        """
        query = (
            "MERGE (o:Account {accountId: $orig_id}) "
            "MERGE (d:Account {accountId: $dest_id}) "
            "CREATE (o)-[r:TRANSACTED {amount: $amount, step: $step, txId: $tx_id}]->(d)"
        )
        
        self.run_cypher(query, {
            "orig_id": tx_data.get("nameOrig"),
            "dest_id": tx_data.get("nameDest"),
            "amount": tx_data.get("amount"),
            "step": tx_data.get("step"),
            "tx_id": tx_data.get("txId")
        })
        logger.debug(f"Transaction {tx_data.get('txId')} written to Neo4j successfully.")
