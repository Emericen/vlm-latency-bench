# vLLM WebSocket Server

Custom vLLM server with WebSocket support for Qwen2.5-VL-7B-Instruct model.

## Quick Start

```bash
make build && make run
```

## Commands

- **Start**: `make build && make run`
- **Logs**: `make logs` 
- **Stop**: `make stop`
- **Test**: `python test.py`
- **Clean**: `make clean`

## Endpoints

- HTTP API: `http://localhost:8000/v1`
- WebSocket API: `ws://localhost:8001`

---

## Alternative: Official vLLM Docker

Makefile using vllm docker. Nothing else required:

```
# vLLM Inference Server Makefile
.PHONY: help build run stop logs clean

# Variables
DOCKER_IMAGE = vllm/vllm-openai:v0.10.0
CONTAINER_NAME = vllm-server
PORT = 8000

# Default target
help:
	@echo "Available targets:"
	@echo "  build  - Build the Docker image"
	@echo "  run    - Run the container (detached)"
	@echo "  stop   - Stop the running container"
	@echo "  logs   - Show container logs (follow)"
	@echo "  clean  - Remove container and image"
	@echo "  help   - Show this help message"

# Build the Docker image
build:
	@echo "Building Docker image: $(DOCKER_IMAGE)"
	docker build -t $(DOCKER_IMAGE) .

# Run the container
run:
	@echo "Starting vLLM inference server..."
	@echo "Server will be available at: http://localhost:$(PORT)"
	@if [ "$$(docker ps -aq -f name=$(CONTAINER_NAME))" ]; then \
		echo "Removing existing container..."; \
		docker rm -f $(CONTAINER_NAME); \
	fi
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p $(PORT):$(PORT) \
		--ipc=host \
		--shm-size=16g \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		$(DOCKER_IMAGE) \
		--model Qwen/Qwen2.5-VL-7B-Instruct \
		--host 0.0.0.0 \
		--port $(PORT) \
		--trust-remote-code
	@echo "Container started. Use 'make logs' to see output."

# Stop the container
stop:
	@echo "Stopping vLLM server..."
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker stop $(CONTAINER_NAME); \
		echo "Container stopped."; \
	else \
		echo "Container is not running."; \
	fi

# Show logs
logs:
	@echo "Following logs for $(CONTAINER_NAME)..."
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker logs -f $(CONTAINER_NAME); \
	else \
		echo "Container is not running."; \
	fi

# Clean up
clean: stop
	@echo "Cleaning up..."
	@if [ "$$(docker ps -aq -f name=$(CONTAINER_NAME))" ]; then \
		docker rm $(CONTAINER_NAME); \
		echo "Container removed."; \
	fi
	@if [ "$$(docker images -q $(DOCKER_IMAGE))" ]; then \
		docker rmi $(DOCKER_IMAGE); \
		echo "Image removed."; \
	fi
```