MODEL      ?= Qwen/Qwen2.5-VL-3B-Instruct
TOOL_CALL_PARSER ?= hermes
GPUS       ?= all
HF_CACHE   ?= $(HOME)/.cache/huggingface

pull:
	docker pull vllm/vllm-openai:latest

run: stop
	docker run -d --name vllm \
	  --gpus $(GPUS) \
	  --restart unless-stopped \
	  -p 8000:8000 \
	  -v $(HF_CACHE):/root/.cache/huggingface \
	  --shm-size=16g --ipc=host \
	  vllm/vllm-openai:latest \
	  --model "$(MODEL)" \
	  --host 0.0.0.0 --port 8000 \
	  --served-model-name "$(MODEL)" \
	  --trust-remote-code \
	  --dtype auto \
	  --enable-prefix-caching \
	  --enable-auto-tool-choice \
	  --tool-call-parser $(TOOL_CALL_PARSER)
	@echo "HTTP API: http://localhost:8000/v1   (health: /v1/models)"

restart: stop run  

stop:
	-@docker rm -f vllm 2>/dev/null || true

logs:
	docker logs -f --tail=200 vllm

ps:
	docker ps --filter name=vllm

health:
	curl -sf http://localhost:8000/v1/models | jq . || curl -sf http://localhost:8000/v1/models

shell:
	docker exec -it vllm /bin/bash

clean: stop
	-@docker rmi vllm/vllm-openai:latest 2>/dev/null || true

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk 'BEGIN{FS=":.*?##"}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'