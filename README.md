# TinyNanoGPT - C++ Port

A pure C++ implementation of a GPT-style language model, ported from the Python TinyNanoGPT example. This implementation uses only standard C++ and no external deep learning libraries.

## Overview

This project implements a transformer-based language model (GPT) in C++17. It includes:
- Token and position embeddings
- Transformer blocks with multi-head self-attention and feed-forward networks
- Layer normalization
- Language model head with weight tying
- Training loop with AdamW optimizer
- Text generation with temperature and top-k sampling

## Model Architecture

The model follows the standard GPT architecture:
- **Embedding Layer**: Token embeddings + Position embeddings
- **Transformer Blocks**: Each block consists of:
  - Multi-head self-attention
  - Feed-forward network (2-layer MLP with GELU activation)
  - Residual connections and layer normalization
- **Output Layer**: Language model head (tied weights with token embedding)

## Building

### Requirements
- C++17 compiler (GCC, Clang, or MSVC)
- Make or your preferred build system

### Build Instructions
```bash
# Compile the training executable
g++ -std=c++17 -O2 -o train train.cpp -I.

# Compile the generation executable
g++ -std=c++17 -O2 -o generate generate.cpp -I.
```

## Usage

### Training
```bash
./train --data ./data/shakespeare.txt --max_steps 1000 --batch_size 32 --block_size 128
```

#### Training Arguments
- `--data`: Path to training text file (default: ./data/shakespeare.txt)
- `--max_steps`: Number of training steps (default: 1000)
- `--batch_size`: Batch size (default: 32)
- `--block_size`: Sequence length (default: 128)
- Other model hyperparameters are fixed in the code (vocab_size=65, n_layer=6, n_head=6, n_embd=384)

### Generation
```bash
./generate --prompt "ROMEO: " --checkpoint ./checkpoint_final.bin --max_new_tokens 200 --temperature 0.8 --top_k 200
```

#### Generation Arguments
- `--prompt`: Initial text to condition on (default: "To be or not")
- `--checkpoint`: Path to model checkpoint file (default: ./checkpoint_final.bin)
- `--max_new_tokens`: Number of tokens to generate (default: 200)
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_k`: Top-k sampling filter (default: 40, set to 0 to disable)
- `--seed`: Random seed (default: -1 for random seed)

## Data Preparation

The model expects raw text files. The training script will:
1. Read the text file
2. Build a vocabulary from unique characters
3. Convert text to token indices
4. Train on batches of sequences

Place your text file in the `data/` directory or specify a path with `--data`.

## Model Checkpoints

Checkpoints are saved periodically during training and include:
- Model weights
- Optimizer state
- Vocabulary mappings (stoi and itos)
- Model configuration

The final checkpoint is saved as `checkpoint_final.bin` in the current directory.

## Implementation Details

### Tensor Operations
All tensor operations are implemented in the `ops` namespace in `model.h`:
- Basic arithmetic: add, sub, mul, div
- Matrix multiplication: matmul
- Layer normalization: layer_norm
- Softmax: softmax
- Dropout: dropout (not used in this implementation for determinism)
- Embedding lookup: embedding
- Transpose: transpose2d

### Model Files
- `model.h`: Contains the Tensor class, model definition (GPT, Block, etc.), and all operations
- `train.cpp`: Training loop with learning rate scheduling and validation
- `generate.cpp`: Text generation with checkpoint loading and sampling

## Performance

This implementation is designed for clarity and correctness rather than maximum performance. For larger models or longer training runs, consider:
- Increasing optimization level (-O3)
- Using compiler-specific optimizations
- Implementing CUDA kernels for GPU acceleration (not included)

## License

This project is for educational purposes. See the original TinyNanoGPT for licensing information.

## Acknowledgements

Based on the TinyNanoGPT Python implementation by Andrej Karpathy and others in the nanoGPT repository.