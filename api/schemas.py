from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


class TransactionInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "TransactionID": "TXN-20240101-0001",
                    "card1": 14204,
                    "TransactionAmt": 117.50,
                    "TransactionDT": 86400,
                    "ProductCD": "W",
                    "card4": "visa",
                    "card6": "credit",
                    "P_emaildomain": "gmail.com",
                    "R_emaildomain": "gmail.com",
                    "C1": 1.0,
                    "C2": 1.0,
                    "D1": 14.0,
                    "vesta_features": [0.0] * 339,
                }
            ]
        }
    )
    TransactionID: str = Field(
        ..., description="Unique transaction identity token identifier."
    )
    card1: int = Field(
        ..., description="Unique card or user account account identifier token."
    )
    TransactionAmt: float = Field(
        ..., description="Transaction amount value in regional currency.", ge=0.0
    )
    TransactionDT: int = Field(
        ..., description="Time delta delta-offset tracker timestamp in seconds."
    )
    ProductCD: str = Field(
        ...,
        description="Product category categorical variable type indicator code mapping.",
    )
    card4: str = Field(
        ..., description="Card network brand marker identification label."
    )
    card6: str = Field(
        ...,
        description="Card application accounting specification layer classification type.",
    )
    P_emaildomain: Optional[str] = Field(
        "UNKNOWN", description="Purchaser domain name structure registry identity."
    )
    R_emaildomain: Optional[str] = Field(
        "UNKNOWN", description="Recipient domain name structure registry identity."
    )

    # Baseline GNN network context vectors
    C1: float = Field(
        0.0, description="Counting dynamic rate aggregation historical metric feature."
    )
    C2: float = Field(
        0.0, description="Counting dynamic rate aggregation historical metric feature."
    )
    D1: float = Field(
        0.0,
        description="Timedelta accounting step measurement reference window feature.",
    )

    # Mock fallback array parameters for Vesta structural alignment vector sizing
    vesta_features: List[float] = Field(
        default_factory=lambda: [0.0] * 339,
        description="Vesta behavior metrics framework fingerprint data placeholder block vector arrays.",
    )


class FraudPredictionResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "transaction_id": "TXN-20240101-0001",
                    "fraud_score": 0.0731,
                    "is_fraudulent": False,
                    "processing_latency_ms": 12.4,
                    "subgraph_nodes": 3,
                    "subgraph_edges": 4,
                    "fallback_used": False,
                    "model_version": "v1.0.0-onnx",
                    "decision_reason": "Low-risk structural transaction topology",
                }
            ]
        }
    )
    transaction_id: str
    fraud_score: float = Field(
        ...,
        description="Continuous probability matrix confidence mapping (0.0 to 1.0).",
    )
    is_fraudulent: bool = Field(
        ...,
        description="Crisp operational transaction routing execution decision binary classification.",
    )
    processing_latency_ms: float = Field(
        ...,
        description="Internal system server inference latency tracking performance benchmark.",
    )
    subgraph_nodes: int = Field(
        default=1,
        description="Total nodes in the localized k-hop neighborhood graph evaluated by the GNN.",
    )
    subgraph_edges: int = Field(
        default=0,
        description="Total topological edges evaluated during GraphSAGE message passing.",
    )
    fallback_used: bool = Field(
        default=False,
        description="Flag indicating if the circuit breaker fell back to the tabular risk engine.",
    )
    model_version: str = Field(
        default="v1.0.0-onnx",
        description="Semantic version string of the production model artifact.",
    )
    decision_reason: str = Field(
        default="Normal transaction pattern",
        description="Human-interpretable decision rationale for compliance review.",
    )


class SystemHealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_version: str
    graph_store: dict
    circuit_breaker: dict
    uptime_seconds: float


class ABPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    transaction_id: str
    ab_group: str = Field(..., description="A/B test group routed to: 'A' or 'B'.")
    model_used: str = Field(
        ..., description="Name of the model that evaluated the transaction."
    )
    fraud_score: float = Field(
        ..., description="Fraud risk score probability (0.0 to 1.0)."
    )
    is_fraudulent: bool = Field(..., description="Decision binary classification.")
    processing_latency_ms: float = Field(
        ..., description="Server inference latency in milliseconds."
    )
