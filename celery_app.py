import os
from celery import Celery

# Redis同时做Broker和Backend
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "vidriatta",
    broker=REDIS_URL,
    backend=REDIS_URL,
)
# Celery基本可靠性设置
celery_app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # 任务不被某个worker一次性抢太多
    broker_transport_options={"visibility_timeout": 3600},  # 单位：秒
    result_expires=86400,  # 结果保存一天
    include=["tasks"],  # 自动发现 tasks.py
)