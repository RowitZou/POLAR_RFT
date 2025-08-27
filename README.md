# POLAR-RFT: Reinforcement Fine-tuning with POLAR Reward Model

This repository contains scripts and utilities for performing reinforcement fine-tuning (RFT) using the POLAR reward model.

## Overview

POLAR-RFT integrates the [POLAR reward model](https://github.com/InternLM/POLAR) with [VERL](https://github.com/volcengine/verl) framework to enable efficient RL fine-tuning of large language models.

## Features

- **PPO Training**: Complete PPO training pipeline with actor-critic setup
- **POLAR Integration**: Built-in support for POLAR reward model as external reward function
- **Multi-node Training**: Distributed training support with Ray
- **Flexible Data Pipeline**: Support for various datasets and custom data formats

## Quick Start

### 1. Environment Setup

Please refer to the [VERL official installation guide](https://github.com/volcengine/verl) for detailed environment setup instructions.

**Important Version Requirements:**

For the inference backend, we recommend using:
- **vLLM 0.8.3**
- **Transformers 4.50.3**

⚠️ **Note**: We have found that higher versions of transformers can cause training instability. Please ensure you use transformers 4.50.3 for optimal performance.

### 2. POLAR Reward Model Deployment

Deploy the POLAR reward model following the instructions in the [official POLAR repository](https://github.com/InternLM/POLAR).

Update the server configuration in `src/polar/reward_func.py`:

```python
# Config reward model server
ADDRESS = "your_server_ip:port"  # Modify according to your server address
SERVER_TYPE = "sglang"  # Options: "sglang", "vllm", "lmdeploy"
MODEL_PATH = "internlm/POLAR-7B"
```

### 3. Data Preparation

Prepare your training data in Parquet format. You can use the provided data preprocessing scripts:

```bash
# Example: Process HH-RLHF dataset
python examples/data_preprocess/full_hh_rlhf.py --local_dir ~/data/hh_rlhf
```

### 4. Configure Training Script

An example of training script `examples/ppo/qwen2_5-7b_hh-rlhf.sh`:

```bash
# Model paths - Update these to your actual model paths
actor_path=Qwen/Qwen2.5-7B-Instruct
critic_path=Qwen/Qwen2.5-7B-Instruct

# Data paths - Update these to your prepared data
train_data_path=/your/path/to/data_train.parquet
test_data_path=/your/path/to/data_test.parquet

# Training parameters
nodes=1                   # Number of nodes
train_batch_size=512      # Training batch size
actor_lr=1e-6             # Actor learning rate
critic_lr=1e-5            # Critic learning rate
```

### 5. Run Training

```bash
bash examples/ppo/qwen2_5-7b_hh-rlhf.sh
```

## Data Format

Training data should be in Parquet format with the following structure:
```python
{
    "data_source": "dataset_name",
    "prompt": [{"role": "user", "content": "..."}, ...],
    "ability": "alility_type",
    "reward_model": {
        "style": "polar",
        "ground_truth": [{"role": "assistant", "content": "..."}]
    }
    "extra_info": {
        # The same as prompt. The purpose is for compatibible usage of verl and polar.
        "prompt": [{"role": "user", "content": "..."}, ...],
    }
}
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Related Projects

- [POLAR](https://github.com/InternLM/POLAR) - The reward model used in this project
- [VERL](https://github.com/volcengine/verl) - The reinforcement learning framework
