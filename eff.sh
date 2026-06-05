#!/bin/bash
# Paper Table 2 (TinyLlama + Qwen2.5-1.5B + Llama3.2-3B LoRA): regular vs Centrifuge
# python efficiency_benchmark.py --suite table2

# Quick smoke (one model, regular + loss_only + centrifuge):
python efficiency_benchmark.py --suite quick \
  --cuda-devices 0,1,2,3 \
  --model TinyLlama/TinyLlama_v1.1 \
  --context-length 2048 \
  --drop-rate 0.5

# Figure 5a — context length sweep:
# python efficiency_benchmark.py --suite fig5a --cuda-devices 0,1,2,3 --model TinyLlama/TinyLlama_v1.1

# Figure 5b — filtering ratio sweep:
# python efficiency_benchmark.py --suite fig5b --cuda-devices 0,1,2,3 --model TinyLlama/TinyLlama_v1.1 --context-length 2048

# Single run:
# python efficiency_benchmark.py --suite single --mode centrifuge --cuda-devices 0,1,2,3 --model TinyLlama/TinyLlama_v1.1
