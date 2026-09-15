FROM python:3.11-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-demucs.txt .
RUN pip install --no-cache-dir -r requirements-demucs.txt

RUN pip install --no-cache-dir yt-dlp numpy soundfile

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
