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

## Contributing

This project is used for learning purposes but improvements are welcome. Please open an issue or pull request if you find a bug.

## Code Style Suggestions

Below are a few ideas for making the code easier to read and extend:

- Introduce type hints to clarify the expected shapes of tensors.
- Break down long functions into smaller helpers where possible.
- Document the expected observation and action spaces for each environment.
- Consider adopting a consistent naming convention for variables and classes.
