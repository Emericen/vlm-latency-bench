FROM python:3.10-slim

# Install system dependencies including C++ compiler
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies using uv
RUN uv pip install --system -r requirements.txt

# Install additional dependencies for OpenAI client
RUN uv pip install --system openai

# Copy application files
COPY main.py .


# Expose ports for vLLM HTTP API and WebSocket
EXPOSE 8000 8001

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting vLLM server..."\n\
vllm serve ${MODEL} \\\n\
    --host ${HOST} \\\n\
    --port ${PORT} \\\n\
    --served-model-name ${MODEL} \\\n\
    --trust-remote-code &\n\
\n\
echo "Waiting for vLLM server to start..."\n\
while ! curl -f http://localhost:${PORT}/health > /dev/null 2>&1; do\n\
    sleep 2\n\
done\n\
\n\
echo "vLLM server is ready, starting WebSocket proxy..."\n\
python main.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]