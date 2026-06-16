FROM python:3.11-slim

WORKDIR /app

# 先装依赖，利用层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# 默认启动 FastAPI（compose 里 ui 服务会覆盖为 streamlit）
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
