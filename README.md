# DeepRewire
DeepRewire is a PyTorch-based project designed to simplify the creation and optimization of neural networks with the concepts from the [Deep Rewiring](https://arxiv.org/abs/1711.05136) paper. ⚠️ Note: The implementation is still under development. Please double-check everything before use.

## Overview

DeepRewire provides tools to convert standard neural network parameters into a rewireable form that can be optimized using the DEEPR and SoftDEEPR algorithms. This allows for maintaining network sparsity during training.

## Features

- **Conversion Functions**: Convert networks to and from rewireable forms.
- **Optimizers**: Use DEEPR and SoftDEEPR to optimize sparse networks.
- **Examples**: Run provided examples to see the conversion and optimization in action.

## Example Usage:
```python
import torch
from deep_rewire import convert_to_deep_rewireable, convert_from_deep_rewireable, SoftDEEPR

# Define your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
)

# Convert model parameters to rewireable form
rewireable_params, other_params = convert_to_deep_rewireable(model)

# Define optimizers
optim1 = SoftDEEPR(rewireable_params, lr=0.05, l1=1e-5) 
optim2 = torch.optim.SGD(other_params, lr=0.05) # Optional, for other parameters

# ... Standard training loop ...

# Convert back to standard form
convert_from_deep_rewireable(model)
# Model has the same parameters but is now sparse.
```

![SoftDEEPR Performance](https://github.com/LuggiStruggi/DeepRewire/blob/main/images/mnist_softdeepr.svg)


## Functionality

### Conversion Functions

#### convert_to_deep_rewireable
```python
convert_to_deep_rewireable(module: nn.Module, handle_biases: str = "second_bias",
                           active_probability: float = None, keep_signs: bool = False)
```
Converts a PyTorch module into a rewireable form.

- **Parameters**:
    - `module` (nn.Module): The model to convert.
    - `handle_biases` (str): Strategy to handle biases. Options are ignore, as_connections, and second_bias.
    - `active_probability` (float): Initial active probability for connections.
    - `keep_signs` (bool): Retain initial network signs for pretrained networks.

 #### convert_from_deep_rewireable
```python
convert_from_deep_rewireable(module: nn.Module)
```
Converts a rewireable module back into its original form, making its sparsity visible.

- **Parameters**:
    - `module` (nn.Module): The model to convert.

### Optimizers

#### DEEPR
```python
DEEPR(params, nc=required, lr, l1, reset_val, temp)
```
The `DEEPR` algorithm keeps a fixed number of connections, which when becoming inactive, new connections are activated randomly to keep the same connectivity.

- `nc` (int): Fixed number of active connections.
- `lr` (float): Learning rate.
- `l1` (float): L1 regularization term.
- `reset_val` (float): Value for newly activated parameters.
- `temp` (float): Temperature affecting noise magnitude.

#### SoftDEEPR
```python
SoftDEEPR(params, lr=0.05, l1=1e-5, temp=None, min_weight=None)
```

The `SoftDEEPR` algorithm has no fixed amount of connections, but also adds noise to its inactive connections to randomly activate them.

- `lr` (float): Learning rate.

- `l1` (float): L1 regularization term.

- `temp` (float): Temperature affecting noise magnitude.

- `min_weight` (float): Minimum value for inactive parameters.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Guillaume Bellec, David Kappel, Wolfgang Maass, Robert Legenstein for their work on Deep Rewiring.

For more details, refer to the [Deep Rewiring paper](https://arxiv.org/abs/1711.05136).
