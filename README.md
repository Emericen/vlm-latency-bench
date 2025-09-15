# vLLM Inference Server

Simplified vLLM Docker setup with prefix caching, tool calling, and quantization options.

## Quick Start

```bash
make pull && make run
```

## Configuration

```bash
# Variables (can be overridden)
MODEL=Qwen/Qwen2.5-VL-3B-Instruct
TOOL_CALL_PARSER=hermes
DTYPE=auto                    # auto, bfloat16, half, float16, float32
QUANTIZATION=                 # fp8, awq, gptq, gguf (empty = no quantization)
GPUS=all
```

## Usage Examples

```bash
# Default configuration
make run

# Use bfloat16 precision
make run DTYPE=bfloat16

# Use FP8 quantization
make run QUANTIZATION=fp8

# Use different model with half precision and AWQ quantization
make run MODEL=meta-llama/Llama-3.1-8B-Instruct DTYPE=half QUANTIZATION=awq
```

## Commands

- **Run**: `make run` - Start server with current config
- **Stop**: `make stop` - Stop the server
- **Logs**: `make logs` - View server logs
- **Status**: `make ps` - Check container status
- **Health**: `make health` - Test API health
- **Shell**: `make shell` - Access container shell
- **Clean**: `make clean` - Remove container and image

## Features

- **Prefix Caching**: Enabled for faster repeated prompts
- **Tool Calling**: OpenAI-compatible function calling with hermes parser
- **Quantization**: Support for FP8, AWQ, GPTQ, GGUF
- **Precision Control**: auto, bfloat16, half, float16, float32
- **GPU Support**: Automatic GPU detection and allocation

## API Endpoint

- **HTTP API**: `http://localhost:8000/v1`
- **Health Check**: `http://localhost:8000/v1/models`