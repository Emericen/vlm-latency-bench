MODEL      ?= Qwen/Qwen2.5-VL-7B-Instruct
TOOL_CALL_PARSER ?= hermes
DTYPE      ?= auto # auto, bfloat16, half, float16, float32
QUANTIZATION ?= # fp8, awq, gptq, gguf (empty = no quantization)
TENSOR_PARALLEL ?= 1 # number of GPUs for tensor parallelism
GPUS       ?= all
HF_CACHE   ?= $(HOME)/.cache/huggingface
PORT       ?= 8000

pull:
	docker pull vllm/vllm-openai:latest

run: stop
	docker run -d --name vllm \
	  --gpus $(GPUS) \
	  --restart unless-stopped \
	  -p $(PORT):8000 \
	  -v $(HF_CACHE):/root/.cache/huggingface \
	  --shm-size=16g --ipc=host \
	  vllm/vllm-openai:latest \
	  --model "$(MODEL)" \
	  --host 0.0.0.0 --port 8000 \
	  --served-model-name "$(MODEL)" \
	  --trust-remote-code \
	  --dtype $(DTYPE) \
	  $(if $(QUANTIZATION),--quantization $(QUANTIZATION)) \
	  --tensor-parallel-size $(TENSOR_PARALLEL) \
	  --enable-prefix-caching \
	  --enable-auto-tool-choice \
	  --tool-call-parser $(TOOL_CALL_PARSER)
	@echo "HTTP API: http://localhost:$(PORT)/v1   (health: /v1/models)"

restart: stop run  

stop:
	-@docker rm -f vllm 2>/dev/null || true

logs:
	docker logs -f --tail=200 vllm

shell:
	docker exec -it vllm /bin/bash

clean: stop
	-@docker rmi vllm/vllm-openai:latest 2>/dev/null || true

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk 'BEGIN{FS=":.*?##"}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'