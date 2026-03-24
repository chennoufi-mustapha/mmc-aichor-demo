import os
import time
import shutil
import tempfile
import boto3
from tensorboardX import SummaryWriter


def dummy_tb_write(message: str = None):
    """
    Hybrid TensorBoard writer:
    - PVC mode  -> write directly to mounted path
    - S3 mode   -> write locally then upload
    - Fallback  -> if S3 fails, copy to PVC fallback
    """

    log_path = os.environ.get("AICHOR_TENSORBOARD_PATH")
    fallback_path = os.environ.get("AICHOR_TENSORBOARD_PVC_PATH")
    aichor_message = os.environ.get("AICHOR_EXPERIMENT_MESSAGE", "")

    if not log_path:
        print("❌ AICHOR_TENSORBOARD_PATH not set")
        return

    # Merge messages
    if message:
        message = f"{message} - {aichor_message}"
    else:
        message = aichor_message or "no message"

    # Detect mode
    is_s3 = log_path.startswith("s3://")
    is_pvc = log_path.startswith("/")

    # Create run folder
    run_id = f"run_{int(time.time())}"

    # -------------------------
    # 1️⃣ WRITE LOGS
    # -------------------------
    if is_pvc:
        logdir = os.path.join(log_path, run_id)
        os.makedirs(logdir, exist_ok=True)
        print(f"[DEBUG] PVC mode → {logdir}")
    else:
        logdir = tempfile.mkdtemp(prefix="tb_logs_")
        print(f"[DEBUG] Temp local write → {logdir}")

    writer = SummaryWriter(logdir)

    writer.add_scalar("debug/value", 1.0, 0)
    writer.add_text("debug/message", message, 0)

    writer.close()
    print("✅ TensorBoard logs written")

    # -------------------------
    # 2️⃣ PVC MODE → DONE
    # -------------------------
    if is_pvc:
        print(f"✅ Logs available at PVC: {logdir}")
        return

    # -------------------------
    # 3️⃣ S3 MODE
    # -------------------------
    if is_s3:
        print(f"[DEBUG] S3 upload → {log_path}")

        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=os.environ.get("AWS_ENDPOINT_URL")
            )

            # Parse path
            s3_path = log_path.replace("s3://", "")
            parts = s3_path.split("/", 1)

            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""

            # 🚨 Validate bucket
            if not bucket or ":" in bucket:
                raise ValueError(f"Invalid bucket name: {bucket}")

            prefix = f"{prefix.rstrip('/')}/{run_id}"

            # Upload files
            for root, _, files in os.walk(logdir):
                for f in files:
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, logdir)
                    key = f"{prefix}/{rel_path}"

                    print(f"[DEBUG] Upload {full_path} → s3://{bucket}/{key}")

                    with open(full_path, "rb") as data:
                        s3.upload_fileobj(data, bucket, key)

            print(f"✅ S3 upload complete → s3://{bucket}/{prefix}")
            return

        except Exception as e:
            print(f"❌ S3 upload failed: {e}")

    # -------------------------
    # 4️⃣ FALLBACK TO PVC
    # -------------------------
    if fallback_path:
        fallback_dir = os.path.join(fallback_path, run_id)
        os.makedirs(fallback_dir, exist_ok=True)

        print(f"[DEBUG] Fallback → PVC: {fallback_dir}")

        for root, _, files in os.walk(logdir):
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(fallback_dir, f)

                print(f"[DEBUG] Copy {src} → {dst}")
                shutil.copy2(src, dst)

        print(f"✅ Logs saved to fallback PVC: {fallback_dir}")
    else:
        print("❌ No fallback PVC configured")

    # Cleanup temp dir (optional)
    try:
        shutil.rmtree(logdir)
    except Exception:
        pass
