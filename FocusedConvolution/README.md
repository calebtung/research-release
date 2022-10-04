# Install instructions
```
conda install torch
conda install torchvision
conda install tqdm
```

# Run instructions
`python focusedconv.py` to run a complete speed+accuracy test experiment. Measure using a power meter to check energy usage.

# Usage instructions
You can `import focusedconv` in your Python code to access the Focused Convolution and use it as a replacement for `torch.nn.Conv2d`.
