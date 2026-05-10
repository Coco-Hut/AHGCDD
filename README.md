# Anchor-guided Hypergraph Condensation with Dual-level Discrimination (ICML2026)
This is the official PyTorch implementation of AHGCDD, an efficient hypergraph condensation framework based on anchor-guided hyperedge synthesis and multi-level discriminative optimization.

## Environment Requirement

Hardware environment: Intel(R) Xeon(R) Silver 4208 CPU, a Quadro RTX 6000 24GB GPU, and 128GB of RAM.

Software environment:Python 3.9.12, Pytorch 1.13.0, and CUDA 11.2.0.

## Quick Start

To run hypergraph condensation on Cora with AHGCDD,

```bash
python main.py --dname=cora --cond_method=ahgcdd
```