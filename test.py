import torch
print(torch.cuda.is_available())  # Should print True if CUDA is available
print(torch.cuda.get_device_name(0))  # Should print the name of your GPU
