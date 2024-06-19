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
