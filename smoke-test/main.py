import argparse
import time
import os

from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.pytorch import pytorchop
from src.operators.xgboost import xgboostop
from src.operators.jobset import jobsetop

OPERATOR_TABLE = {
    "ray": rayop,
    "kuberay": rayop,
    "jax": jaxop,
    "pytorch": pytorchop,
    "xgboost": xgboostop,
    "jobset": jobsetop
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AIchor Smoke test on any operator"
    )

    parser.add_argument(
        "--operator",
        type=str,
        default="jobset",
        choices=OPERATOR_TABLE.keys(),
        help="operator name"
    )

    parser.add_argument(
        "--sleep",
        type=int,
        default=0,  # FIX: should be int, not string
        help="sleep time in seconds"
    )

    parser.add_argument(
        "--tb-write",
        action="store_true",  # FIX: correct way to handle boolean flags
        help="test write to tensorboard"
    )

    args = parser.parse_args()

    # Safe environment variable access (prevents crashes if missing)
    print(os.environ.get("runtime_var"))
    print(os.environ.get("AICHOR_USER_NAME"))
    print(os.environ.get("AICHOR_USER_EMAIL"))
    print(os.environ.get("AICHOR_CLUSTER_NAME"))
    print(os.environ.get("VCS_TYPE"))
    print(os.environ.get("AICHOR_INPUT_PATH"))
    print(os.environ.get("AICHOR_OUTPUT_PATH"))
    print(os.environ.get("AICHOR_TENSORBOARD_PATH"))

    print(f"using {args.operator} operator")

    OPERATOR_TABLE[args.operator](args.tb_write)

    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)
