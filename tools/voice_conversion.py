import os
import json
import logging
import time
import redis
from typing import Any
import ast 

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# 导入必要的模块
from tools.vc_api import load_models, voice_conversion, export_audio
import requests
from pydantic import BaseModel

# 日志配置
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis 配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
TASK_QUEUE = os.getenv("TASK_QUEUE", "tasks_queue")

# 重试配置
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # 秒

class TaskPayload(BaseModel):
    user_id: str
    task_id: str
    task_time: str
    reference_audio_path: str
    source_audio_path: str
    callback_url: str

def send_callback(url: str, payload: dict, max_attempts: int = MAX_RETRY_ATTEMPTS) -> bool:
    """带重试机制的回调通知"""
    for attempt in range(max_attempts):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.warning(f"Callback attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                logger.error("All callback attempts failed")
                return False
            time.sleep(RETRY_DELAY)

def process_task(payload: TaskPayload, models: Any):
    """任务处理"""
    try:
        MODEL, SEMANTIC_FN, VOCODER_FN, CAMPPLUS_MODEL, TO_MEL, MEL_FN_ARGS, DEVICE, SR, HOP_LENGTH = models

        output_dir = os.path.dirname(payload.reference_audio_path)
        output_file = os.path.join(output_dir, f"{payload.task_id}_generated.mp3")

        logger.info(f"Processing task {payload.task_id}...")
        sr, generated_wave_chunks = voice_conversion(
            source=payload.source_audio_path,
            target=payload.reference_audio_path,
            diffusion_steps=25,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            model=MODEL,
            semantic_fn=SEMANTIC_FN,
            vocoder_fn=VOCODER_FN,
            campplus_model=CAMPPLUS_MODEL,
            to_mel=TO_MEL,
            mel_fn_args=MEL_FN_ARGS,
            device=DEVICE,
            sr=SR,
            hop_length=HOP_LENGTH
        )

        export_audio(
            wave_chunks=generated_wave_chunks,
            sr=sr,
            bitrate="320k",
            format="mp3",
            output_file=output_file
        )

        logger.info(f"Task {payload.task_id} completed. Output saved to {output_file}")

        # 发送成功回调
        callback_payload = {
            "status": "completed",
            "task_id": payload.task_id,
            "output_file": output_file
        }
        send_callback(payload.callback_url, callback_payload)

    except Exception as e:
        logger.error(f"Task {payload.task_id} failed: {e}", exc_info=True)
        
        # 发送失败回调
        callback_payload = {
            "status": "failed",
            "task_id": payload.task_id,
            "error": str(e)
        }
        send_callback(payload.callback_url, callback_payload)

def main():
    """主任务处理循环"""
    # 加载模型
    logger.info("Loading models...")
    models = load_models()
    logger.info("Models loaded successfully.")

    # 创建 Redis 连接
    redis_client = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=REDIS_DB
    )

    logger.info(f"Starting task worker. Listening on queue: {TASK_QUEUE}")

    while True:
        try:
            # 从 Redis 队列中获取任务
            task_data = redis_client.lpop(TASK_QUEUE)
            
            if task_data:
                try:
                    # 解析任务负载
                    task_data_dict = ast.literal_eval(task_data.decode("utf-8"))
                    task_payload = TaskPayload(**task_data_dict)
                    
                    # 处理任务
                    process_task(task_payload, models)
                
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error: {je}")
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
            
            else:
                # 没有任务时稍微休眠，避免持续高CPU占用
                time.sleep(1)
        
        except redis.exceptions.ConnectionError:
            logger.error("Redis connection lost. Retrying...")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Unexpected error in task worker: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()