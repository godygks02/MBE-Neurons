import torch
import torch.nn.functional as F
from Ternary_EA_MBE_neurons import TernaryEAMBENeuron

x = torch.linspace(-10, 10, 1000).unsqueeze(1)
y = F.gelu(x)

mbe = TernaryEAMBENeuron(num_basis=4, timesteps=16, dt=1.0, alpha=10.0)
y_pred = mbe(x)
loss = F.mse_loss(y_pred, y)
loss.backward()

for name, p in mbe.named_parameters():
    grad_norm = p.grad.norm().item() if p.grad is not None else 0.0
    print(f"{name}: {grad_norm}")
