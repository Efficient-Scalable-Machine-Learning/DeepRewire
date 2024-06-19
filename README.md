# DeepRewire
Some PyTorch stuff to easily create and optimize networks as in the Deep Rewiring paper (https://igi-web.tugraz.at/PDF/241.pdf).
⚠️ Not completely sure if correctly implemented. Please double check everything.
Run examples like `python -m examples.example1` from `src`.

Usage example:
```python
model = someTorchModule()
rewireable_params, other_params = convert_to_deep_rewireable(model)
optim1 = SoftDEEPR(sparse_params, lr=0.05, l1=1e-5) 
optim2 = torch.optim.SGD(lr=0.05) # not needed if no other parameters

# [... STANDARD TRAINING LOOP ...]

convert_from_deep_rewireable(model)
# model has same paramters as before but is hopefully sparse now.
```

![SoftDEEPR Performance](https://github.com/LuggiStruggi/DeepRewire/blob/main/images/mnist_softdeepr.svg)


## Functionality

### Conversion

```python
convert_to_deep_rewireable(module: nn.Module, handle_biases: str = "second_bias",
                           active_probability: float = None, keep_signs: bool = False)
```
Using `convert_to_deep_rewireable` you can convert a PyTorch Module into a form that can be optimized by the `(Soft)DEEPR` algorithm.
The function returns two lists. First a list of parameters that can be optimized with `(Soft)DEEPR` and then a list for which you can use any other optimizer.

- `handle_biases` selects the strategy how to handle biases. You can currently choose between
    - `ignore`: Ignores the bias as a parameter for `(Soft)DEEPR` and adds it to the list of other parameters
    - `as_connections`: Just converts it as every other connection, randomly assigning one fixed sign
    - `second_bias`: Splits it into two biases: one with negative and one with positive sign, such that it can be optimized by `(Soft)DEEPR` directly, without fixing the sign.
 
- `active_probability`: Sets the probability of an connection being initially active. Per default the connection is set to active based on the weights of the given module.

- `keep_signs`: Keeps the initially given network functionally as is. This is interesting when using an already pretrained network.

```python
convert_from_deep_rewireable(module: nn.Module)
```
Using `convert_from_deep_rewireable` you can convert a module from the rewireable form back into its initial form, where you can actually see its sparsity.

### Optimizers
