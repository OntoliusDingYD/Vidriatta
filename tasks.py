import os
import json
import tempfile
import boto3
import botocore
from ultralytics import YOLO
from celery_app import celery_app

BUCKET_NAME = os.getenv("BUCKET_NAME", "vidriatta-detection")

# boto3 带重试的 Session/Client
session = boto3.session.Session()
config = botocore.config.Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60,
)
s3 = session.client("s3", config=config)

# 在worker进程启动时加载模型（避免每个任务重复加载）
# 注：如用 GPU，建议控制并发/单实例独占GPU
MODEL_PATH = os.getenv("MODEL_PATH", "./models/yolov8n.pt")
model = YOLO(MODEL_PATH)

@celery_app.task(bind=True, autoretry_for=(Exception,), retry_backoff=True, retry_jitter=True, max_retries=3)
def run_detection(self, input_key: str, output_key: str):
    """
    S3下载->YOLO->本地生成结果->上传S3->返回检测结构
    """
    with tempfile.TemporaryDirectory() as td:
        local_in = os.path.join(td, "in.jpg")
        local_out = os.path.join(td, "out.jpg")

        # 下载原图
        s3.download_file(BUCKET_NAME, input_key, local_in)

        # 推理
        results = model(local_in)
        results[0].save(filename=local_out)

        # 解析检测框
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "conf": round(float(box.conf[0]), 4),
                "bbox": [round(float(x), 2) for x in box.xyxy[0]],
            })

        # 上传结果图
        s3.upload_file(local_out, BUCKET_NAME, output_key)

        return {
            "output_s3_key": output_key,
            "detections": detections,
        }