from locust import HttpUser, task, between
import random

class FraudApiUser(HttpUser):
    wait_time = between(0.1, 1.0)

    @task(20)
    def predict_transaction(self):
        payload = {
            "TransactionID": f"T{random.randint(1000, 999999)}",
            "card1": random.randint(1000, 18000),
            "TransactionAmt": round(random.uniform(1.0, 5000.0), 2),
            "TransactionDT": random.randint(86400, 31536000),
            "ProductCD": random.choice(["W", "C", "H", "R", "S"]),
            "card4": random.choice(["visa", "mastercard", "american express", "discover"]),
            "card6": random.choice(["credit", "debit"]),
            "vesta_features": [random.uniform(-5, 5) for _ in range(339)],
            "C1": random.uniform(0, 10),
            "C2": random.uniform(0, 10),
            "D1": random.uniform(0, 50)
        }
        self.client.post("/api/v1/predict", json=payload)

    @task(1)
    def health_check(self):
        self.client.get("/health")
