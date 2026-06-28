import os
import json
import shutil
import logging
from datetime import datetime
import argparse

logger = logging.getLogger("ModelRegistry")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class ModelRegistry:
    def __init__(
        self, registry_dir="artifacts/registry", prod_dir="artifacts/production"
    ):
        self.registry_dir = registry_dir
        self.prod_dir = prod_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        os.makedirs(self.prod_dir, exist_ok=True)

    def _copy_version_files(self, version_dir):
        files_to_copy = ["fraud_model.onnx", "scaler.pkl", "prob_calibrator.pkl"]
        for f in files_to_copy:
            src = os.path.join(version_dir, f)
            dst = os.path.join(self.prod_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                logger.warning(f"File {f} not found in {version_dir}")

    def promote_to_production(self, version: str):
        """Copies specified version to artifacts/production/"""
        version_dir = os.path.join(self.registry_dir, version)
        if not os.path.exists(version_dir):
            raise ValueError(f"Version {version} does not exist in registry.")

        # Read current active version
        active_meta_path = os.path.join(self.prod_dir, "active_version.json")
        old_version = "None"
        if os.path.exists(active_meta_path):
            with open(active_meta_path, "r") as f:
                old_version = json.load(f).get("version", "None")

        self._copy_version_files(version_dir)

        meta_path = os.path.join(version_dir, "metadata.json")
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, active_meta_path)

        logger.info(f"Promoted {version} to production. (Previous: {old_version})")

    def rollback(self, version: str, reason: str = "Manual rollback"):
        """Restores any previous version to production"""
        logger.info(f"Initiating rollback to {version}. Reason: {reason}")
        self.promote_to_production(version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Registry CLI")
    subparsers = parser.add_subparsers(dest="command")

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("version", help="Version to promote (e.g., v1)")

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("version", help="Version to rollback to (e.g., v1)")
    rollback_parser.add_argument(
        "--reason", default="Manual rollback", help="Reason for rollback"
    )

    args = parser.parse_args()
    registry = ModelRegistry()

    if args.command == "promote":
        registry.promote_to_production(args.version)
    elif args.command == "rollback":
        registry.rollback(args.version, args.reason)
    else:
        parser.print_help()
