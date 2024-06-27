# DeepRewire
DeepRewire is a PyTorch-based project designed to simplify the creation and optimization of sparse neural networks with the concepts from the [Deep Rewiring](https://arxiv.org/abs/1711.05136) paper by Bellec et. al. ⚠️ Note: This implementation is not made by any of the authors. Please double-check everything before use.

## Overview

DeepRewire provides tools to convert standard neural network parameters into a form that can be optimized using the DEEPR and SoftDEEPR optimizers. This allows for gaining network sparsity during training.

## Installation

Install using `pip install deep_rewire`

## Features

- **Conversion Functions**: Convert networks to and from rewireable forms.
- **Optimizers**: Use DEEPR and SoftDEEPR to optimize sparse networks.
- **Examples**: Run provided examples to see the conversion and optimization in action.

## Example Usage:
```python
import torch
from deep_rewire import convert, reconvert, SoftDEEPR

# Define your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Convert model parameters to rewireable form
rewireable_params, other_params = convert(model)

# Define optimizers
optim1 = SoftDEEPR(rewireable_params, lr=0.05, l1=1e-5) 
optim2 = torch.optim.SGD(other_params, lr=0.05) # Optional, for parameters that are not rewireable

# ... Standard training loop ...

# Convert back to standard form
reconvert(model)
# Model has the same parameters but is now (hopefully) sparse.
```
**examples/softdeepr.py:**
![SoftDEEPR Performance](https://github.com/LuggiStruggi/DeepRewire/blob/main/images/mnist_softdeepr3.svg)


## Functionality

### Conversion Functions

#### convert
```python
deep_rewire.convert(module: nn.Module, handle_biases: str = "second_bias",
                           active_probability: float = None, keep_signs: bool = False)
```
Converts a PyTorch module into a rewireable form.

- **Parameters**:
    - `module` (nn.Module): The model to convert.
    - `handle_biases` (str): Strategy to handle biases. Options are 'ignore', 'as_connections', and 'second_bias'.
    - `active_probability` (float): Probability for connections to be active right after conversion.
    - `keep_signs` (bool): Retain initial network signs and start with all connections active (for pretrained networks).

 #### reconvert



```python
deep_rewire.reconvert(module: nn.Module)
```
Converts a rewireable module back into its original form, making its sparsity visible.

- **Parameters**:
    - `module` (nn.Module): The model to convert.

### Optimizers

#### DEEPR
```python
deep_rewire.DEEPR(params, nc=required, lr, l1, reset_val, temp)
```
The `DEEPR` algorithm keeps a fixed number of connections, which when becoming inactive, new connections are activated randomly to keep the same connectivity.

- `nc` (int): Fixed number of active connections.
- `lr` (float): Learning rate.
- `l1` (float): L1 regularization term.
- `reset_val` (float): Value for newly activated parameters.
- `temp` (float): Temperature affecting noise magnitude.

#### SoftDEEPR
```python
deep_rewire.SoftDEEPR(params, lr=0.05, l1=1e-5, temp=None, min_weight=None)
```

The `SoftDEEPR` algorithm has no fixed amount of connections, but also adds noise to its inactive connections to randomly activate them.

- `lr` (float): Learning rate.

- `l1` (float): L1 regularization term.

- `temp` (float): Temperature affecting noise magnitude.

- `min_weight` (float): Minimum value for inactive parameters.

#### SoftDEEPRWrapper
```python
deep_rewire.SoftDEEPRWrapper(params, base_optim, l1=1e-5, temp=None, min_weight=None, **optim_kwargs)
```

Uses the `SoftDEEPR` algorithm regarding keeping the connections sparse but updates the parameters using any chosen torch optimizer (SGD, Adam..).

- `base_optim` (torch.optim.Optimizer): The basic optimizer to use for updating the parameters

- `l1` (float): L1 regularization term.

- `temp` (float): Temperature affecting noise magnitude.

- `min_weight` (float): Minimum value for inactive parameters.

- `**optim_kwargs`: Arguments for the base optimizer 

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements and fix my mistakes :).

## License
This project is licensed under the MIT License.

## Acknowledgements
- Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein for their paper on Deep Rewiring.

For more details, refer to their [Deep Rewiring paper](https://arxiv.org/abs/1711.05136) or their [TensorFlow tutorial](https://github.com/guillaumeBellec/deep_rewiring).
