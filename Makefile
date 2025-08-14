DOCKER_IMAGE = vllm-websocket
CONTAINER_NAME = vllm-websocket-server
PORT = 8000
MODEL = Qwen/Qwen2.5-VL-3B-Instruct

.PHONY: build run logs stop clean restart

build:
	@echo "Building custom vLLM WebSocket image..."
	docker build -t $(DOCKER_IMAGE) .

run:
	@echo "Starting vLLM server with WebSocket support..."
	@echo "HTTP API available at: http://localhost:$(PORT)/v1"
	@echo "WebSocket API available at: ws://localhost:8001"
	@if [ "$$(docker ps -aq -f name=$(CONTAINER_NAME))" ]; then \
		echo "Removing existing container..."; \
		docker rm -f $(CONTAINER_NAME); \
	fi
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus all \
		-p $(PORT):$(PORT) \
		-p 8001:8001 \
		--ipc=host \
		--shm-size=16g \
		-v ~/.cache/huggingface:/root/.cache/huggingface \
		-e MODEL=$(MODEL) \
		-e HOST=0.0.0.0 \
		-e PORT=$(PORT) \
		$(DOCKER_IMAGE)
	@echo "Container started. Use 'make logs' to see output."

logs:
	@echo "Following logs for $(CONTAINER_NAME)..."
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker logs -f $(CONTAINER_NAME); \
	else \
		echo "Container is not running."; \
	fi

stop:
	@echo "Stopping vLLM WebSocket server..."
	@if [ "$$(docker ps -q -f name=$(CONTAINER_NAME))" ]; then \
		docker stop $(CONTAINER_NAME); \
		echo "Container stopped."; \
	else \
		echo "Container is not running."; \
	fi

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

restart: stop run