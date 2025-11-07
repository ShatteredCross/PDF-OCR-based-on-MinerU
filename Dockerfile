# 使用 PyTorch 官方镜像
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖（poppler-utils 用于 PDF 处理）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8100

# 启动应用
CMD ["python", "web_demo.py"]