import time
import random
import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"

def generate_mock_transaction():
    """
    Creates random transaction details matching schemas.TransactionRequest Pydantic structure.
    """
    tx_id = random.randint(100000, 999999)
    step = random.randint(1, 100)
    amount = round(random.uniform(10.0, 50000.0), 2)
    name_orig = f"C{random.randint(10000, 99999)}"
    old_bal_orig = round(random.uniform(amount, amount * 10), 2)
    new_bal_orig = round(old_bal_orig - amount, 2)
    name_dest = f"M{random.randint(10000, 99999)}"
    old_bal_dest = round(random.uniform(0, 10000), 2)
    new_bal_dest = round(old_bal_dest + amount, 2)
    
    # 5% chance of adversarial disguise injection (creates an outlier signature)
    is_adversarial = random.random() < 0.05
    if is_adversarial:
        amount = round(random.uniform(100000, 500000), 2)
        old_bal_orig = 0.0
        new_bal_orig = 0.0
        logger.info(f"[ADVERSARIAL ATTACK INJECTED] Simulating transaction {tx_id}")
        
    return {
        "txId": tx_id,
        "step": step,
        "amount": amount,
        "nameOrig": name_orig,
        "oldbalanceOrg": old_bal_orig,
        "newbalanceOrig": new_bal_orig,
        "nameDest": name_dest,
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest
    }

def run_stream(tps: int = 5):
    """
    Continuously feeds transactions into the API gateway at a designated transactions-per-second.
    """
    logger.info(f"Starting mock transaction stream targeting {API_URL} at {tps} TPS...")
    delay = 1.0 / tps
    
    success_count = 0
    fail_count = 0
    
    while True:
        payload = generate_mock_transaction()
        try:
            start_time = time.perf_counter()
            response = requests.post(API_URL, json=payload, timeout=2.0)
            latency = (time.perf_counter() - start_time) * 1000.0
            
            if response.status_code == 200:
                res_data = response.json()
                status = "🚨 FLAGGED" if res_data.get("flagged") else "✅ PASS"
                logger.info(
                    f"Tx {payload['txId']}: {status} | Score: {res_data['risk_score']:.3f} | Latency: {latency:.1f}ms"
                )
                success_count += 1
            else:
                logger.error(f"Failed response: {response.status_code} - {response.text}")
                fail_count += 1
        except Exception as e:
            logger.error(f"Network error targeting API gateway: {e}")
            fail_count += 1
            
        time.sleep(delay)

if __name__ == "__main__":
    import sys
    target_tps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    try:
        run_stream(tps=target_tps)
    except KeyboardInterrupt:
        logger.info("Transaction stream stopped by keyboard interrupt.")
