FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Tránh tương tác với user trong quá trình build
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirements trước để tận dụng cache của Docker
COPY requirements.txt .

# Cài đặt dependencies Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ source code
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p /app/data /app/weights /app/results

# Expose port
EXPOSE 8000

# Set Python path
ENV PYTHONPATH=/app

# Khởi động mặc định: API service
CMD ["python3", "-m", "cli", "serve", "--host", "0.0.0.0", "--port", "8000"] 