#!/usr/bin/env python3
"""
Generate a Dockerfile for running vLLM on DGX Spark.
Lists available models and creates appropriate configuration.
"""

import os
import sys

# DGX Spark Supported Models
SUPPORTED_MODELS = [
    {"name": "GPT-OSS-20B", "quantization": "MXFP4", "hf_handle": "openai/gpt-oss-20b"},
    {"name": "GPT-OSS-120B", "quantization": "MXFP4", "hf_handle": "openai/gpt-oss-120b"},
    {"name": "Llama-3.1-8B-Instruct", "quantization": "FP8", "hf_handle": "nvidia/Llama-3.1-8B-Instruct-FP8"},
    {"name": "Llama-3.1-8B-Instruct", "quantization": "NVFP4", "hf_handle": "nvidia/Llama-3.1-8B-Instruct-FP4"},
    {"name": "Llama-3.3-70B-Instruct", "quantization": "NVFP4", "hf_handle": "nvidia/Llama-3.3-70B-Instruct-FP4"},
    {"name": "Qwen3-8B", "quantization": "FP8", "hf_handle": "nvidia/Qwen3-8B-FP8"},
    {"name": "Qwen3-8B", "quantization": "NVFP4", "hf_handle": "nvidia/Qwen3-8B-FP4"},
    {"name": "Qwen3-14B", "quantization": "FP8", "hf_handle": "nvidia/Qwen3-14B-FP8"},
    {"name": "Qwen3-14B", "quantization": "NVFP4", "hf_handle": "nvidia/Qwen3-14B-FP4"},
    {"name": "Qwen3-32B", "quantization": "NVFP4", "hf_handle": "nvidia/Qwen3-32B-FP4"},
    {"name": "Qwen2.5-VL-7B-Instruct", "quantization": "NVFP4", "hf_handle": "nvidia/Qwen2.5-VL-7B-Instruct-FP4"},
    {"name": "Phi-4-multimodal-instruct", "quantization": "FP8", "hf_handle": "nvidia/Phi-4-multimodal-instruct-FP8"},
    {"name": "Phi-4-multimodal-instruct", "quantization": "NVFP4", "hf_handle": "nvidia/Phi-4-multimodal-instruct-FP4"},
    {"name": "Phi-4-reasoning-plus", "quantization": "FP8", "hf_handle": "nvidia/Phi-4-reasoning-plus-FP8"},
    {"name": "Phi-4-reasoning-plus", "quantization": "NVFP4", "hf_handle": "nvidia/Phi-4-reasoning-plus-FP4"},
    {"name": "Nemotron3-Nano", "quantization": "BF16", "hf_handle": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"},
    {"name": "Nemotron3-Nano", "quantization": "FP8", "hf_handle": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"},
]

# Default vLLM version
DEFAULT_VLLM_VERSION = "25.11-py3"
# Special version required for Nemotron3-Nano
NEMOTRON_VLLM_VERSION = "25.12.post1-py3"


def list_models():
    """Display all available models."""
    print("\n" + "=" * 80)
    print("DGX Spark Supported Models for vLLM")
    print("=" * 80)
    print(f"\n{'#':<4} {'Model Name':<30} {'Quantization':<12} {'HF Handle'}")
    print("-" * 80)

    for idx, model in enumerate(SUPPORTED_MODELS, 1):
        note = " [*]" if "Nemotron" in model["name"] else ""
        print(f"{idx:<4} {model['name']:<30} {model['quantization']:<12} {model['hf_handle']}{note}")

    print("-" * 80)
    print("[*] Nemotron3-Nano requires special vLLM version: " + NEMOTRON_VLLM_VERSION)
    print()


def get_vllm_version(model_name: str) -> str:
    """Get the appropriate vLLM version for a model."""
    if "Nemotron" in model_name:
        return NEMOTRON_VLLM_VERSION
    return DEFAULT_VLLM_VERSION


def generate_dockerfile(model_hf_handle: str, vllm_version: str, output_path: str = "Dockerfile"):
    """Generate a Dockerfile for the selected model."""
    dockerfile_content = f"""# vLLM Dockerfile for DGX Spark
# Model: {model_hf_handle}
# vLLM Version: {vllm_version}

FROM nvcr.io/nvidia/vllm:{vllm_version}

# Set environment variables
ENV MODEL_NAME="{model_hf_handle}"
ENV PORT=8000

# Expose the API port
EXPOSE $PORT

# Default command to serve the model
CMD ["sh", "-c", "vllm serve $MODEL_NAME --port $PORT"]
"""

    with open(output_path, "w") as f:
        f.write(dockerfile_content)

    return output_path


def generate_instructions(model_hf_handle: str, vllm_version: str) -> str:
    """Generate instructions for building and running the container."""
    return f"""
================================================================================
                        BUILD AND RUN INSTRUCTIONS
================================================================================

1. Build the Docker image:
   docker build -t vllm-server .

2. Run the container:
   docker run -it --gpus all -p 8000:8000 vllm-server

   Or run directly without building (using the base image):
   docker run -it --gpus all -p 8000:8000 \\
       nvcr.io/nvidia/vllm:{vllm_version} \\
       vllm serve "{model_hf_handle}"

3. Test the server:
   curl http://localhost:8000/v1/chat/completions \\
       -H "Content-Type: application/json" \\
       -d '{{
           "model": "{model_hf_handle}",
           "messages": [{{"role": "user", "content": "Hello!"}}],
           "max_tokens": 500
       }}'

4. Cleanup:
   docker rm $(docker ps -aq --filter ancestor=vllm-server)
   docker rmi vllm-server

================================================================================
"""


def main():
    print("\n" + "=" * 80)
    print("           vLLM Dockerfile Generator for DGX Spark")
    print("=" * 80)

    # List available models
    list_models()

    # Get user selection
    while True:
        try:
            selection = input("Select a model number (1-{}), or 'q' to quit: ".format(len(SUPPORTED_MODELS)))

            if selection.lower() == 'q':
                print("Exiting.")
                sys.exit(0)

            model_idx = int(selection) - 1
            if 0 <= model_idx < len(SUPPORTED_MODELS):
                break
            else:
                print(f"Please enter a number between 1 and {len(SUPPORTED_MODELS)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Get selected model
    selected_model = SUPPORTED_MODELS[model_idx]
    model_name = selected_model["name"]
    model_hf_handle = selected_model["hf_handle"]
    quantization = selected_model["quantization"]

    # Determine vLLM version
    vllm_version = get_vllm_version(model_name)

    print(f"\nSelected: {model_name} ({quantization})")
    print(f"HuggingFace Handle: {model_hf_handle}")
    print(f"vLLM Version: {vllm_version}")

    if "Nemotron" in model_name:
        print("\n** Note: Using special vLLM version for Nemotron3-Nano support **")

    # Generate Dockerfile
    output_dir = os.path.dirname(os.path.abspath(__file__))
    dockerfile_path = os.path.join(output_dir, "Dockerfile")
    generate_dockerfile(model_hf_handle, vllm_version, dockerfile_path)

    print(f"\nDockerfile generated: {dockerfile_path}")

    # Print instructions
    instructions = generate_instructions(model_hf_handle, vllm_version)
    print(instructions)

    # Also save instructions to a file
    instructions_path = os.path.join(output_dir, "INSTRUCTIONS.txt")
    with open(instructions_path, "w") as f:
        f.write(f"Model: {model_name} ({quantization})\n")
        f.write(f"HuggingFace Handle: {model_hf_handle}\n")
        f.write(f"vLLM Version: {vllm_version}\n")
        f.write(instructions)

    print(f"Instructions saved to: {instructions_path}")


if __name__ == "__main__":
    main()
