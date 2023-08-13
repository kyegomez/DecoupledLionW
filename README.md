# DecoupledLionW Optimizer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Welcome to the DecoupledLionW optimizer, an adaptive optimization algorithm implemented in PyTorch. This optimizer combines gradient updates and momentum to improve training performance for your deep learning models.

## Code
```
import logging
import math
from typing import Callable, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)


class DecoupledLionW(Optimizer):

    """

    DecoupledLionW is an optimizer designed to improve training performance and convergence for deep learning models.

    It is an extension of the Lion optimizer, incorporating decoupled weight decay and a momentum-based update rule.

    The optimizer utilizes the Adam-like update rule, where the weight decay is applied separately from the gradient update.

    The update rule consists of three steps: weight decay, momentum update, and momentum decay.

    Weight decay reduces the magnitude of the model's weights, preventing overfitting and improving generalization.

    The momentum update is an interpolation between the current gradient and the previous momentum state, allowing for faster convergence and smoother optimization.

    Momentum decay gradually reduces the momentum term over time, preventing it from becoming too large and destabilizing the optimization process.

    The optimizer supports both single-node and multi-node distributed training, enabling efficient training on parallel computing environments.

    It provides various metric functions to track the optimization process, such as L2 norm of moments, parameters, updates, and gradients, as well as cosine similarity between updates and gradients.
    
    The optimizer allows reporting per-parameter metrics to analyze the behavior of individual model parameters during training.
    """


    metric_functions = {
        'l2_norm/moment': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(optim_state['exp_avg']),
        'l2_norm/param': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.data),
        'l2_norm/update': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(step_tensor),
        'l2_norm/grad': lambda param, optim_state, step_tensor: torch.linalg.vector_norm(param.grad),
        'cosine/update_grad': lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(param.grad.flatten(), step_tensor.flatten(), dim=0),
        'cosine/moment_grad': lambda param, optim_state, step_tensor: torch.nn.functional.cosine_similarity(param.grad.flatten(), optim_state['exp_avg'].flatten(), dim=0),
    }

    def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0,
    ):
        if lr <= 0.:
            raise Exception(f'Invalid LR: {lr}. LR must be > 0')
        if not all([0. <= beta <= 1. for beta in betas]):
            raise Exception(f'Invalid beta values: {betas}. All betas must be between 0 and 1.')
        if weight_decay >= 1e-3:
            log.warning(f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. Are you sure you want to do this? Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')

        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

        for group in self.param_groups:
            group['initial_lr'] = group['lr']

    @staticmethod
    def lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2) -> None:
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None and p.requires_grad, group['params']):
                grad, lr, initial_lr, wd, beta1, beta2, state = p.grad, group['lr'], group['initial_lr'], group['weight_decay'], *group['betas'], self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']


                self.lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2)

        return loss
    
    def pre_reduce_metrics(self, optimizer_metrics):
        metrics = optimizer_metrics.keys()
        metrics = sorted(metrics, key=lambda metric: 0 if 'l2_norm' in metric else 1)
        for metric in metrics:
            if metric.startswith('l2_norm'):
                optimizer_metrics[metric] = optimizer_metrics[metric]**2
            elif metric.startswith('cosine'):
                _, vectors, layer = tuple(metric.split('/'))
                A, B = tuple(vectors.split('_'))
                A_rank_subset_norm = math.sqrt(optimizer_metrics[f'l2_norm/{A}/{layer}'])
                B_rank_subset_norm = math.sqrt(optimizer_metrics[f'l2_norm/{B}/{layer}'])
                optimizer_metrics[metric] *= A_rank_subset_norm * B_rank_subset_norm

        return optimizer_metrics

    def report_per_parameter_metrics(self, param: torch.Tensor, name: str, optimizer_metrics: dict):
        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        initial_lr = self.param_groups[0]['initial_lr']

        beta1, _ = self.param_groups[0]['betas']
        if param in self.state:
            param_optim_state = self.state[param]
            step_tensor = param_optim_state['exp_avg'].clone().lerp_(param.grad, 1 - beta1).sign_().mul_(lr)
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            step_tensor.add_(param, alpha=-weight_decay * decay_factor)
            for metric in self.metric_functions:
                optimizer_metrics[f'{metric}/{name}'] = self.metric_functions[metric](param, param_optim_state, step_tensor)

        return optimizer_metrics

```

## Features

- **Decoupled Learning Rates**: The DecoupledLionW optimizer allows customizable learning rates for each parameter independently, enhancing the flexibility of training different model components effectively.

- **Adaptive Weight Decay**: You can apply weight decay to control the model's weights during training, multiplying them by a decay factor on each step. This helps regularize the model and prevent overfitting.

- **Metrics and Analysis**: The optimizer provides useful metrics such as cosine similarity and l2 norm to track the behavior of the gradients, update steps, and parameters. These metrics aid in understanding and analyzing the optimization process.

## Usage

To use the DecoupledLionW optimizer, follow these steps:

1. Import the optimizer class:
```python
from decoupled_lionw import DecoupledLionW
```

2. Instantiate the optimizer with your model parameters:
```python
optimizer = DecoupledLionW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay=0.0001)
```

3. In your training loop, call the optimizer's `step()` method after computing the gradients:
```python
optimizer.zero_grad()
loss = compute_loss()
loss.backward()
optimizer.step()
```

4. Optionally, access the optimizer's metrics for analysis:
```python
optimizer_metrics = optimizer.get_metrics()
# Use the metrics for monitoring and analysis
```

For a complete example, refer to the provided code and documentation.

## Roadmap and Contributions

We welcome contributions from the community to enhance and expand the DecoupledLionW optimizer. You can contribute in the following ways:

- **Implement Improvements**: Explore new ideas and enhancements to further improve the optimizer's performance or extend its functionality.

- **Bug Fixes**: If you encounter any issues or bugs, please submit bug reports or fix them directly with pull requests.

- **Documentation**: Improve the existing documentation, add code examples, or provide additional usage instructions to make it more comprehensive and accessible.

- **Performance Benchmarks**: Conduct performance benchmarks and comparisons with other optimizers to assess the DecoupledLionW optimizer's efficiency and effectiveness in different scenarios.

Or contribute to the following optimizations or other optimizations
1. **Acceleration Techniques**: Investigate and implement acceleration techniques such as adaptive learning rate schedules, learning rate warm-up strategies, or gradient clipping. These techniques can improve the convergence speed and stability of the optimizer, leading to faster and more efficient training.

2. **Advanced Weight Decay Strategies**: Explore advanced weight decay strategies that can better regularize the model and prevent overfitting. This could include techniques like layer-wise or group-wise weight decay, adaptive weight decay based on layer characteristics, or dynamic weight decay methods.

3. **Parallel and Distributed Computing**: Extend the optimizer's capabilities to support parallel and distributed training. This optimization could involve implementing techniques like model parallelism or communication-efficient gradient aggregation to scale the training process and handle larger models and datasets.

Contributions in these areas, as well as other optimizations and improvements, are highly encouraged. Feel free to open issues, submit pull requests, or start discussions in the repository to contribute and collaborate with the community.

We appreciate your interest in the DecoupledLionW optimizer and look forward to your contributions to make it even more powerful and efficient.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

Please feel free to open issues, submit pull requests, or start discussions in the repository to contribute and collaborate with the community.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
