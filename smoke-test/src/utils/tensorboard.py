import os
import tempfile
import boto3
from tensorboardX import SummaryWriter


def dummy_tb_write(message: str):
    # Get S3 path
    log_path = os.environ.get("AICHOR_TENSORBOARD_PATH")
    if log_path is None:
        print('"AICHOR_TENSORBOARD_PATH" env var not found')
        return

    # Get experiment message
    aichor_message = os.environ.get("AICHOR_EXPERIMENT_MESSAGE")

    if message is None:
        message = aichor_message
    else:
        message = f"{message} - {aichor_message}"

    # 1️⃣ Write locally (safe)
    local_logdir = tempfile.mkdtemp(prefix="tb_logs_")
    print(f"[DEBUG] Writing TensorBoard logs locally to: {local_logdir}")

    writer = SummaryWriter(local_logdir)

    # ✅ IMPORTANT: write scalar so TensorBoard shows dashboard
    writer.add_scalar("debug/value", 1.0, 0)

    # optional text
    writer.add_text("debug/message", message, 0)

    writer.close()

    # 2️⃣ Upload to S3
    print(f"[DEBUG] Uploading logs to: {log_path}")

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
    )

    # Parse s3://bucket/path
    s3_path = log_path.replace("s3://", "")
    bucket = s3_path.split("/")[0]
    prefix = "/".join(s3_path.split("/")[1:]).rstrip("/")

    # Upload all files preserving structure
    for root, _, files in os.walk(local_logdir):
        for f in files:
            full_path = os.path.join(root, f)

            # preserve folder structure
            rel_path = os.path.relpath(full_path, local_logdir)
            key = f"{prefix}/{rel_path}"

            print(f"[DEBUG] Uploading {full_path} -> s3://{bucket}/{key}")

            with open(full_path, "rb") as data:
                s3.upload_fileobj(data, bucket, key)

    print(f"✅ TensorBoard logs successfully uploaded to {log_path}")
