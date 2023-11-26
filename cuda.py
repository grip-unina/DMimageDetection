"""
This module provides utilities for interacting with CUDA-capable GPUs 
using PyTorch. It includes functions to identify and select GPU devices 
for computations, allowing users to leverage GPU acceleration for their 
PyTorch models. Key functionalities include:

- Checking the index of the currently selected CUDA device.
- Determining the total number of available CUDA devices (GPUs).
- Retrieving the name of a specific CUDA device.
- Dynamically setting the computation device to GPU if available, 
otherwise using CPU.
- If running on a CUDA device, additional information such as 
device name and memory usage (allocated and cached) is displayed.

This module is particularly useful for scenarios where performance 
optimization is crucial and the presence of a GPU can significantly 
speed up computations.

Functions:
- torch.cuda.current_device(): Returns the index of the currently 
selected CUDA device.
- torch.cuda.device_count(): Returns the number of available 
CUDA devices.
- torch.cuda.get_device_name(device_index): Returns the name of the 
specified CUDA device.
- torch.device(): Sets the device for computation to GPU if available, 
else CPU.
- Additional CUDA information display when using a GPU, including 
device name, allocated memory, and cached memory.

Example Usage:
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        # Additional CUDA-specific information
"""

import torch

# Get index of currently selected device
torch.cuda.current_device()  # returns 0 in my case

# Get number of GPUs available
torch.cuda.device_count()  # returns 1 in my case

# Get the name of the device
torch.cuda.get_device_name(0)  # good old Tesla K80

# Setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()

# Additional Info when using cuda
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_cached(0) / 1024**3, 1), "GB")
