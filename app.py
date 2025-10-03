import os
import io
import uuid
import hashlib
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import boto3
import botocore
import redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from celery_app import celery_app
from tasks import run_detection

# ------- 基本配置 -------
BUCKET_NAME = os.getenv("BUCKET_NAME", "vidriatta-detection")
UPLOAD_PREFIX = os.getenv("UPLOAD_PREFIX", "uploads/")
RESULT_PREFIX = os.getenv("RESULT_PREFIX", "results/")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
app.config["JSON_AS_ASCII"] = False

# S3 Client（带重试）
session = boto3.session.Session()
config = botocore.config.Config(
    retries={"max_attempts": 5, "mode": "standard"},
    connect_timeout=5,
    read_timeout=60,
)
s3 = session.client("s3", config=config)

# Redis：做幂等/缓存
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    storage_uri=os.getenv("LIMITER_STORAGE_URI", "redis://localhost:6379/1"),  # 用 1 号库做限流
    strategy="moving-window",  # 可选：滑动窗口算法
    default_limits=["60 per minute", "5 per second"],  # 默认限流
)
rdb = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# key 约定：哈希 -> 结果缓存
def _hash_key(sha):
    return f"img:{sha}:result"     # 值存 JSON：{"task_id"或"output_s3_key", "detections"...}
def _task_key(task_id):
    return f"task:{task_id}:sha"   # task_id -> sha，方便查询/清理
def _inflight_key(sha):    
    return f"img:{sha}:inflight"  # 值: task_id
def _result_key(sha):      
    return f"img:{sha}:result"    # 值: JSON(string)

@app.route("/predict", methods=["POST"])
@limiter.limit("10/minute")
def predict():
    """
    接收图片 -> 计算sha256（幂等去重）-> 如已有缓存直接返回
             -> 否则上传S3 -> 投递Celery -> 返回 task_id
    """
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "No image file provided"}), 400

    # 读取内容到内存并计算 sha256
    buf = file.read()
    sha = hashlib.sha256(buf).hexdigest()

    # 命中缓存：直接返回已生成的结果（免重复推理）
    cache = rdb.get(_hash_key(sha))
    if cache:
        # cache 为 JSON 字符串，直接返回即可（里头包含 output_s3_key/detections）
        return jsonify({"cached": True, **eval(cache)}), 200
    
    # 命中进行中任务：返回 task_id（前端可继续轮询）
    existing_task = rdb.get(_inflight_key(sha))
    if existing_task:
        return jsonify({"task_id": existing_task, "inflight": True}), 202

    # 未命中：生成文件名并上传原图到 S3
    ext = os.path.splitext(secure_filename(file.filename))[1] or ".jpg"
    uid = uuid.uuid4().hex
    input_key = f"{UPLOAD_PREFIX}{uid}{ext}"
    output_key = f"{RESULT_PREFIX}result_{uid}{ext}"

    # 回卷 buffer 供上传
    s3.upload_fileobj(io.BytesIO(buf), BUCKET_NAME, input_key)

    # 投递异步任务
    task = run_detection.delay(input_key, output_key)

    # 关系映射：task_id -> sha，方便任务完成时回写缓存（可选：在轮询端处理）
    rdb.setex(_inflight_key(sha), 3600, task.id)  # 任务进行中，保留一小时
    rdb.setex(_task_key(task.id), 3600, sha)

    return jsonify({"task_id": task.id}), 202

@app.route("/tasks/<task_id>", methods=["GET"])
def task_status(task_id):
    """
    轮询任务状态；完成则写入缓存并返回
    """
    res = celery_app.AsyncResult(task_id)
    state = res.state

    if state == "PENDING":
        return jsonify({"state": state}), 200
    elif state in ("RETRY", "STARTED"):
        return jsonify({"state": state}), 200
    elif state == "SUCCESS":
        data = res.get()
        sha = rdb.get(_task_key(task_id))
        if sha:
            rdb.setex(_result_key(sha), 24*3600, str(data))   # 结果缓存1天
            rdb.delete(_inflight_key(sha))                    # 清理进行中
        return jsonify({"state": state, **data}), 200
    elif state == "FAILURE":
        return jsonify({"state": state, "error": str(res.info)}), 500
    else:
        return jsonify({"state": state}), 200

@app.route("/healthz", methods=["GET"])
def healthz():
    return "ok", 200