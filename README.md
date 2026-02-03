# vLLM on DGX Spark

## Quick Start with Python Script (Recommended)

Use the interactive Python script to generate a Dockerfile for your chosen model:

```bash
python3 generate_vllm_dockerfile.py
```

The script will:
1. Display all supported DGX Spark models
2. Prompt you to select a model
3. Generate a `Dockerfile` configured for that model
4. Provide build and run instructions

After running the script, build and run your container:

```bash
docker build -t vllm-server .
docker run -it --gpus all -p 8000:8000 vllm-server
```

## Manual Setup

### Pull the vLLM Container

```bash
export LATEST_VLLM_VERSION=25.11-py3
docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
```

For Nemotron3-Nano model support, use version `25.12.post1-py3`:

```bash
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3
```

### Run vLLM

```bash
docker run -it --gpus all -p 8000:8000 \
    nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
    vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct"
```

### Test the Server

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "messages": [{"role": "user", "content": "12*17"}],
        "max_tokens": 500
    }'
```

### Cleanup

```bash
docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
```

## DGX Spark Supported Models
| Model                         | Quantization | Support Status | HF Handle                                           |
|------------------------------|--------------|----------------|-----------------------------------------------------|
| GPT-OSS-20B                  | MXFP4        | ✅             | openai/gpt-oss-20b                                  |
| GPT-OSS-120B                 | MXFP4        | ✅             | openai/gpt-oss-120b                                 |
| Llama-3.1-8B-Instruct        | FP8          | ✅             | nvidia/Llama-3.1-8B-Instruct-FP8                    |
| Llama-3.1-8B-Instruct        | NVFP4        | ✅             | nvidia/Llama-3.1-8B-Instruct-FP4                    |
| Llama-3.3-70B-Instruct       | NVFP4        | ✅             | nvidia/Llama-3.3-70B-Instruct-FP4                   |
| Qwen3-8B                     | FP8          | ✅             | nvidia/Qwen3-8B-FP8                                 |
| Qwen3-8B                     | NVFP4        | ✅             | nvidia/Qwen3-8B-FP4                                 |
| Qwen3-14B                    | FP8          | ✅             | nvidia/Qwen3-14B-FP8                                |
| Qwen3-14B                    | NVFP4        | ✅             | nvidia/Qwen3-14B-FP4                                |
| Qwen3-32B                    | NVFP4        | ✅             | nvidia/Qwen3-32B-FP4                                |
| Qwen2.5-VL-7B-Instruct       | NVFP4        | ✅             | nvidia/Qwen2.5-VL-7B-Instruct-FP4                   |
| Phi-4-multimodal-instruct    | FP8          | ✅             | nvidia/Phi-4-multimodal-instruct-FP8                |
| Phi-4-multimodal-instruct    | NVFP4        | ✅             | nvidia/Phi-4-multimodal-instruct-FP4                |
| Phi-4-reasoning-plus         | FP8          | ✅             | nvidia/Phi-4-reasoning-plus-FP8                     |
| Phi-4-reasoning-plus         | NVFP4        | ✅             | nvidia/Phi-4-reasoning-plus-FP4                     |
| Nemotron3-Nano               | BF16         | ✅             | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16          |
| Nemotron3-Nano               | FP8          | ✅             | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8           |



