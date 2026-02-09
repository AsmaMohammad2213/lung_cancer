FROM python:3.10-slim

WORKDIR /app

# System deps for Pillow / Torch
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*


# Install Python deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


RUN mkdir -p static/uploads

EXPOSE 5000

CMD ["python", "app.py"]
