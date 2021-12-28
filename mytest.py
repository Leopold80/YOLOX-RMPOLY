import torch

smooth_l1 = torch.nn.SmoothL1Loss()

loss = smooth_l1(torch.Tensor([[20], [30]]), torch.Tensor([[30], [40]]))
print(loss)
