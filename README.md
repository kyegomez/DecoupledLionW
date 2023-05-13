# DecoupledLionW Optimizer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Welcome to the DecoupledLionW optimizer, an adaptive optimization algorithm implemented in PyTorch. This optimizer combines gradient updates and momentum to improve training performance for your deep learning models.

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

Please feel free to open issues, submit pull requests, or start discussions in the repository to contribute and collaborate with the community.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
