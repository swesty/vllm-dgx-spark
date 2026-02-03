# vLLM Dockerfile for DGX Spark
# Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
# vLLM Version: 25.12.post1-py3

FROM nvcr.io/nvidia/vllm:25.12.post1-py3

# Set environment variables
ENV MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
ENV PORT=8000

# Expose the API port
EXPOSE $PORT

# Default command to serve the model
CMD ["sh", "-c", "vllm serve $MODEL_NAME --port $PORT"]
