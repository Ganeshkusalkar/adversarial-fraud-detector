import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

class MLTrackingManager:
    """
    Unified manager for W&B and MLflow run configurations and model metrics logging.
    """

    def __init__(self, use_wandb: bool = True, use_mlflow: bool = True, project_name: str = "adversarial-fraud-detector"):
        self.use_wandb = use_wandb and HAS_WANDB
        self.use_mlflow = use_mlflow and HAS_MLFLOW
        self.project_name = project_name

        if not HAS_WANDB and use_wandb:
            logger.warning("Weights & Biases is not installed. Disabling wandb tracking.")
        if not HAS_MLFLOW and use_mlflow:
            logger.warning("MLflow is not installed. Disabling mlflow tracking.")

    def start_run(self, run_name: str, config: Dict[str, Any] = None) -> None:
        """
        Starts a logging run for both platforms.
        """
        logger.info(f"Starting experiment run: {run_name}...")
        
        if self.use_wandb:
            wandb.init(project=self.project_name, name=run_name, config=config)
            
        if self.use_mlflow:
            mlflow.set_experiment(self.project_name)
            mlflow.start_run(run_name=run_name)
            if config:
                mlflow.log_params(config)

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        Log dictionary of metrics at a specific training step.
        """
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
            
        logger.debug(f"Metrics logged at step {step}: {metrics}")

    def end_run(self) -> None:
        """
        Completes the current tracking session.
        """
        if self.use_wandb:
            wandb.finish()
            
        if self.use_mlflow:
            mlflow.end_run()
            
        logger.info("Experiment run finished.")
