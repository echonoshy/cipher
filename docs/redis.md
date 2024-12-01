sudo apt update
sudo apt install redis-server

启动
redis-server


测试：
redis-cli
ping 

开发环境：
redis-server --dir ./databases

<!-- redis-server --port 6381 --bind 0.0.0.0 --dir /data/redis --logfile ./log/redis.log -->
redis-server --dir ./databases --logfile ./logs/redis.log


生产环境：
systemctl start redis 
systemctl enable redis 
systemctl stop / restart redis 


vim /etc/redis/redis.conf

使用logrotate 分割日志



curl -X POST "http://localhost:8000/submit_task/" \
    -F "user_id=12345111" \
    -F "reference_audio=@examples/reference/azuma_0.wav" \
    -F "source_audio=@examples/reference/s1p1.wav" \
    -F "task_id=abc123111" \
    -F "task_time=2024-12-01T12:00:00" \
    -F "callback_url=http://example.com/callback"


拿到redis中所有的任务
LRANGE tasks_queue 0 -1