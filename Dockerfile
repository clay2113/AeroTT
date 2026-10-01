FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py yolov8n-seg.pt ./

EXPOSE 8080
CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-8080}"]
