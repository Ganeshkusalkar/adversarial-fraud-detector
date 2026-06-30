import random
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger("SystemInit")


def seed_everything(seed: int = 42) -> None:
    """
    Forces complete determinism across all underlying matrix backends.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Guarantees reproducible, identical convolution selection behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(f"Global system random seed locked securely to: {seed}")
