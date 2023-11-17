import torch

# Assuming you have a list of PyTorch arrays
list_of_tensors = [torch.randn(1, 133, 3), torch.randn(1, 133, 3), torch.randn(1, 133, 3)]

# Concatenate the list of tensors along a specified dimension
concatenated_tensor = torch.cat(list_of_tensors, dim=0)  # Assuming you want to concatenate along dimension 1

print("Original sizes:", [tensor.size() for tensor in list_of_tensors])
print("Concatenated size:", concatenated_tensor.size())