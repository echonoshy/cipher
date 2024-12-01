from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from redis import Redis
import os

# 初始化 FastAPI 和 Redis
app = FastAPI()
redis_client = Redis(host="localhost", port=6379, db=0)

# 可配置的任务队列名称
TASK_QUEUE = os.getenv("TASK_QUEUE", "tasks_queue")
DATA_DIR = "data"

@app.post("/submit_task/")
async def submit_task(
    user_id: str = Form(...),
    reference_audio: UploadFile = Form(...),
    source_audio: UploadFile = Form(...),
    task_id: str = Form(...),
    task_time: str = Form(...),
    callback_url: str = Form(...)
):
    try:
        # 构建存储路径
        base_dir = os.path.join(DATA_DIR, user_id, task_id)
        os.makedirs(base_dir, exist_ok=True)
        
        reference_audio_path = os.path.join(base_dir, f"reference_{reference_audio.filename}")
        source_audio_path = os.path.join(base_dir, f"source_{source_audio.filename}")
        
        # 写入音频文件
        with open(reference_audio_path, "wb") as ref_file:
            ref_file.write(await reference_audio.read())
        
        with open(source_audio_path, "wb") as src_file:
            src_file.write(await source_audio.read())

        # 构建任务报文
        task_payload = {
            "user_id": user_id,
            "task_id": task_id,
            "task_time": task_time,
            "reference_audio_path": reference_audio_path,
            "source_audio_path": source_audio_path,
            "callback_url": callback_url
        }
        
        # 将任务存入 Redis
        redis_client.rpush(TASK_QUEUE, str(task_payload))

        # 返回任务状态
        return JSONResponse({"status": "Task is being processed", "task_id": task_id})
    
    except Exception as e:
        return JSONResponse({"status": "Error", "message": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task_server:app", host="0.0.0.0", port=8000, reload=True)
