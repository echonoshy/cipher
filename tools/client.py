from flask import Flask, request, render_template, redirect, url_for, jsonify
import os
import requests
from datetime import datetime
from threading import Thread

app = Flask(__name__)

# 存储任务状态
tasks = {}

# 上传目录
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户上传的文件
        reference_audio = request.files['reference_audio']
        source_audio = request.files['source_audio']

        if reference_audio and source_audio:
            # 保存文件
            ref_path = os.path.join(UPLOAD_FOLDER, reference_audio.filename)
            src_path = os.path.join(UPLOAD_FOLDER, source_audio.filename)
            reference_audio.save(ref_path)
            source_audio.save(src_path)

            # 创建任务 ID
            task_id = f"task_{int(datetime.now().timestamp())}"
            tasks[task_id] = {"status": "processing", "result_audio_url": None}

            # 提交任务到服务端
            Thread(target=submit_task, args=(task_id, ref_path, src_path)).start()

            return redirect(url_for('index'))
    
    # 渲染主页面
    return render_template('index.html')

@app.route('/tasks', methods=['GET'])
def get_tasks():
    """返回任务状态的 JSON 数据"""
    return jsonify({"tasks": tasks})

def submit_task(task_id, reference_audio, source_audio):
    """向服务端提交任务"""
    callback_url = "http://localhost:5000/callback"
    data = {
        "user_id": "12345111",
        "task_id": task_id,
        "task_time": datetime.now().isoformat(),
        "callback_url": callback_url
    }
    files = {
        "reference_audio": open(reference_audio, "rb"),
        "source_audio": open(source_audio, "rb")
    }

    try:
        # 替换为服务端实际地址
        response = requests.post("http://localhost:8000/submit_task/", data=data, files=files)
        if response.status_code == 200:
            print(f"任务 {task_id} 提交成功")
        else:
            print(f"任务 {task_id} 提交失败: {response.text}")
            tasks[task_id]['status'] = "failed"
    except Exception as e:
        print(f"任务 {task_id} 提交异常: {e}")
        tasks[task_id]['status'] = "failed"
    finally:
        # 关闭文件句柄
        files["reference_audio"].close()
        files["source_audio"].close()

@app.route('/callback', methods=['POST'])
def callback():
    """接收服务端回调"""
    data = request.json
    task_id = data.get("task_id")
    result_audio_url = "http://localhost:8080/" + data.get("result_audio_url")

    if not task_id or not result_audio_url:
        return {"error": "Invalid callback data"}, 400

    # 更新任务状态
    if task_id in tasks:
        tasks[task_id]['status'] = "processed"
        tasks[task_id]['result_audio_url'] = result_audio_url
        return {"message": "Callback received"}, 200
    else:
        return {"error": "Task ID not found"}, 404

if __name__ == "__main__":
    app.run(debug=True)
