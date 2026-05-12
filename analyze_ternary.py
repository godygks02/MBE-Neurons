import torch
import torch.nn.functional as F
from Ternary_EA_MBE_neurons import train_ternary_mbe_neuron

x = torch.linspace(-10, 10, 5).unsqueeze(1)
target = F.gelu(x)

mbe = train_ternary_mbe_neuron(
    target_func=F.gelu, 
    x_range=(-10, 10), 
    num_samples=1000, 
    num_epochs=10, 
    lr=0.05, 
    num_basis=4, 
    timesteps=16, 
    alpha=10.0
)

mbe.eval()
out, s_seq, d_seq = mbe(x, return_sequences=True)

print("Target:", target.squeeze().tolist())
print("Output:", out.squeeze().tolist())

print("\nSpikes for x=-10:")
for t in range(16):
    print(f"t={t}:", s_seq[t, :, 0, 0].tolist())

print("\nSpikes for x=10:")
for t in range(16):
    print(f"t={t}:", s_seq[t, :, -1, 0].tolist())
