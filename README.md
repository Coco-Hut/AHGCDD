# AHGCDD
This is the official PyTorch implementation of AHGCDD.

## Environment Requirement

Hardware environment: Intel(R) Xeon(R) Silver 4208 CPU, a Quadro RTX 6000 24GB GPU, and 128GB of RAM.

Software environment:Python 3.9.12, Pytorch 1.13.0, and CUDA 11.2.0.

## Quick Start

To run hypergraph condensation on Cora with AHGCDD,

```bash
python main.py --dname=cora --cond_method=ahgcdd
```