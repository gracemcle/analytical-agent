"""
CoT Abstract Reasoning Model for Memory Access

Inputs: 
Device Kernel
Array Size
Block, Grid size
SM Count
Register size
L1/Shared Memory size
L2 size
HBM Size
 |
 V
Memory Analysis Prompt
 |
 V
Synthesis
 |
 V
Runtime Prediction
"""
h100_specs = """
FP64 30 teraFLOPS
FP64 Tensor Core 60 teraFLOPS
FP32 60 teraFLOPS
TF32 Tensor Core 835 teraFLOPS
GPU Memory 94GB
GPU Memory Bandwidth 3.9TB/sec
Interconnect NviidaNVLink 600GB/s
Threads / Warp 32
Max Warps / SM 64
Max Threads / SM 2048
Max Thread Blocks / SM 32
Max 32-bit Registers / SM 65536
Max Registers / THread 255
FP32 Cores / SM
Shared Memory Size / SM 228 kB
SMs 114
L2 Cache size 50MB
"""

import os
import ollama
from ollama import Client

os.environ['NO_PROXY'] = 'zenith.ftpn.ornl.gov,localhost,127.0.0.1'
client = Client(host='http://localhost:11434') # locally hosted ollama server excl

stencil_code = None

with open('gpu_code_examples/tile_stencil.cu', 'r') as f:
    stencil_code = f.read()

# model = "gpt-oss:20b"
model = "gpt-oss:120b"

temperature = .5

system_prompt = "You are a knowledgeable assistant with knowledge about H100 specs, CUDA code, and performance analysis"

specs_context = f"These are the specifications for an Nvidia H100 NVL GPU: {h100_specs}"

source_code_context = f"This is my CUDA code: {stencil_code}"

question_1 = "What is the runtime of this code?"

prompt = f"""
# GPU Specifications
{h100_specs}

# CUDA Kernel Code
```cuda
{stencil_code}
```

# Analysis Request
{question_1}

Please provide a detailed runtime analysis considering memory bandwidth, compute throughput, and occupancy to justify the runtime prediction.
"""

response = client.chat(
    model=model,
    messages=[
        {
            'role': 'system',
            'content': system_prompt
        },
        {
            'role': 'user',
            'content': prompt
        }
    ],
    options={
        'temperature': temperature
    }
)
response = response['message']['content']

print(response)