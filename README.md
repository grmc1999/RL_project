# RL Project

This repository contains a collection of reinforcement learning training routines and neural network models used for a university course project. The code implements agents such as PPO, DDPG, GRPO and an Active Inference based approach.

## Repository Structure

```
models/           # Neural networks and agent utilities
train_routines/   # Training loops for each algorithm
*.ipynb           # Example notebooks
```

## Getting Started

The code assumes a Python 3 environment with `torch` and `einops` installed. You can run any of the training scripts in `train_routines` by creating the corresponding agent and environment.

Example:

```python
from train_routines.PPO import PPO
# create your environment and networks here
trainer = PPO(actor, critic, env, mem_args)
trainer.train(episodes=100)
```



