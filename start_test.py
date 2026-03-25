import torch
import subprocess

# Check CUDA availability and display GPU information
if torch.cuda.is_available():
    print("CUDA is available")
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")

    # Iterate through each GPU device and display detailed information
    for i in range(device_count):
        print(f"\n--- GPU {i} Detailed Information ---")

        # Get device properties using PyTorch
        props = torch.cuda.get_device_properties(i)
        print(f"  Device Name: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        # Convert total memory from bytes to gigabytes for readability
        print(f"  Total Memory: {props.total_memory / (1024 ** 3):.2f} GB")
        print(f"  Multi-processor Count: {props.multi_processor_count}")

        # Attempt to get current memory usage using nvidia-smi (optional)
        try:
            result = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=memory.used',
                '--format=csv,noheader,nounits',
                '-i', str(i)
            ], encoding='utf-8')
            memory_used = int(result.strip())
            print(f"  Used Memory: {memory_used} MB")
        except Exception as e:
            print(f"  Unable to retrieve used memory: {e}")

else:
    print("CUDA is not available, will use CPU")

# Check tree-sitter compatibility for AST parsing
print("Checking tree-sitter compatibility...")
from tree_sitter_languages import get_language, get_parser

# Attempt to get parser and language for C++ as a test
parser = get_parser('cpp')
language = get_language('cpp')

if parser is None or language is None:
    print("tree-sitter compatibility check failed")
else:
    print("tree-sitter compatibility check passed")

# Set HuggingFace mirror endpoint for improved accessibility in certain regions
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Test HuggingFace Hub connectivity by downloading a config file
from huggingface_hub import hf_hub_download
import json

# Download config.json for Qwen3-32B model and extract model type
config_path = hf_hub_download(repo_id='Qwen/Qwen3-32B', filename='config.json')
model_type = json.load(open(config_path, 'r'))['model_type']
print(f"Model type from config: {model_type}")

# Check available model types on HuggingFace Hub
from huggingface_hub import get_model_types

model_types = get_model_types()
print(f"'qwen3' in model types: {'qwen3' in model_types}")

# Check if specific model exists in the list of available models
print(f"'Qwen3-32B' in model list: {'Qwen3-32B' in [model.id for model in model_types]}")