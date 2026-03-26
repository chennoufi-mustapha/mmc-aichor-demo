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
    parser = argparse.ArgumentParser(description='AIchor Smoke test on any operator')
    parser.add_argument("--operator", type=str, default="jobset", choices=OPERATOR_TABLE.keys(), help="operator name")
    parser.add_argument("--sleep", type=int, default=0, help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")
    print(os.environ.get("runtime_var"))
    print(os.environ["AICHOR_USER_NAME"])
    print(os.environ["AICHOR_USER_EMAIL"])
    print(os.environ["AICHOR_CLUSTER_NAME"])
    print(os.environ["VCS_TYPE"])
    print(os.environ["AICHOR_INPUT_PATH"])
    print(os.environ["AICHOR_OUTPUT_PATH"])
    print(os.environ["AICHOR_TENSORBOARD_PATH"])

    args = parser.parse_args()

    print(f"using {args.operator} operator")
    OPERATOR_TABLE[args.operator](args.tb_write)

    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)
