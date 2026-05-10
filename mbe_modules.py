import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from MBE_neurons import MBENeuron, train_mbe_neuron
from approximate_fp_mult import MBEMultiplier
from approximate_layernorm import MBELayerNorm
from approximate_softmax import MBESoftmax

class MBEActivation(nn.Module):
    """Generic MBE-based Activation Function Wrapper."""
    def __init__(self, target_func, x_range=(-5.0, 5.0), num_basis=8, timesteps=16, model_path=None):
        super().__init__()
        self.target_func = target_func
        self.x_range = x_range
        self.num_basis = num_basis
        self.timesteps = timesteps
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            print(f"[MBEActivation] Loading pre-trained model from {model_path}")
            self.mbe = MBENeuron.load(model_path)
        else:
            self.mbe = None

    def fit(self, x_range=None, epochs=5000, lr=0.01):
        if x_range:
            self.x_range = x_range

        # Check if model already exists at model_path
        if self.mbe is None and self.model_path and os.path.exists(self.model_path):
            print(f"[MBEActivation] Loading existing model from {self.model_path}")
            self.mbe = MBENeuron.load(self.model_path)

        if self.mbe is not None:
            # Skip training if already loaded
            return
        
        # Estimate alpha
        temp_x = torch.linspace(self.x_range[0], self.x_range[1], 1000)
        target_y = self.target_func(temp_x)
        alpha = float(target_y.abs().max().item())
        alpha = max(alpha, 1e-3)

        print(f"[MBEActivation] Training for range {self.x_range}, alpha={alpha:.4f}")
        self.mbe = train_mbe_neuron(
            target_func=self.target_func,
            x_range=self.x_range,
            num_samples=10000,
            num_epochs=epochs,
            lr=lr,
            num_basis=self.num_basis,
            timesteps=self.timesteps,
            alpha=alpha
        )
        if self.model_path:
            self.mbe.save(self.model_path)

    def forward(self, x):
        if self.mbe is None:
            return self.target_func(x)
        
        # We only need the output value for activations.
        # return_sequences=False is much faster as it skips sequence tensor creation.
        self.mbe.eval()
        with torch.no_grad():
            out = self.mbe(x, return_sequences=False)
        return out

class MBEGELU(MBEActivation):
    def __init__(self, **kwargs):
        super().__init__(target_func=F.gelu, **kwargs)

class MBELinear(nn.Module):
    """
    MBE-based Linear Layer.
    Approximates y = MBE_FP(W, x) + b
    """
    def __init__(self, in_features, out_features, bias=True, num_basis=8, timesteps=16, epochs=1000, lr=0.01, alpha=1.0, model_path=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.mbe = MBENeuron(num_basis=num_basis, timesteps=timesteps, alpha=alpha)
        self.num_basis = num_basis

    def initialize_multiplier(self, mbe_id_model):
        """Stores the identity MBE neuron for encoding input."""
        self.mbe = mbe_id_model

    def forward(self, x):
        """
        x: (Batch, In)
        Returns: (Batch, Out)
        """
        if self.mbe is None:
            return F.linear(x, self.weight, self.bias)

        # 1. Normalize input x to [-1, 1] for stable MBE encoding
        # Use a safe maximum for normalization
        s_x = x.abs().max().detach().clamp(min=1e-6)
        x_norm = (x / s_x).clamp(-1, 1)

        # 1. Encode input x into spike sequences
        # MBE forward returns (decoded_val, s_seq, d_seq_scaled)
        _, s_seq, d_seq_scaled = self.mbe(x_norm, return_sequences=True)
        
        # 3. Efficient Vectorized Accumulation (SOP Logic)
        T, N, B, I = s_seq.shape
        flat_contributions = (s_seq * d_seq_scaled).reshape(T*N, B, I)
        
        # decoded x_norm: sum_t,n (s_tn * d_tn)
        encoded_input_norm = flat_contributions.sum(dim=0) # (Batch, In)
        
        # Restore scale
        encoded_input = encoded_input_norm * s_x
        
        # 4. Final Linear pass
        out = F.linear(encoded_input, self.weight, self.bias)
        
        return out

    def load_from_standard_linear(self, linear):
        with torch.no_grad():
            self.weight.copy_(linear.weight)
            if self.bias is not None:
                self.bias.copy_(linear.bias)

class MBEConv1D(nn.Module):
    """
    MBE-based Conv1D for Hugging Face GPT-2.
    GPT-2's Conv1D is essentially a Linear layer where weights are (in_features, out_features).
    """
    def __init__(self, nf, nx, num_basis=8, timesteps=16, epochs=1000, lr=0.01, alpha=1.0, model_path=None):
        super().__init__()
        self.nf = nf # out_features
        self.nx = nx # in_features
        self.weight = nn.Parameter(torch.Tensor(nx, nf))
        self.bias = nn.Parameter(torch.Tensor(nf))
        self.mbe = MBENeuron(num_basis=num_basis, timesteps=timesteps, alpha=alpha)
        self.num_basis = num_basis
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.mbe = MBENeuron.load(model_path)

    def initialize_multiplier(self, mbe_id_model):
        self.mbe = mbe_id_model

    def forward(self, x):
        if self.mbe is None:
            size_out = x.size()[:-1] + (self.nf,)
            out = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
            return out.view(*size_out)

        s_x = x.abs().amax(dim=-1, keepdim=True).detach().clamp(min=1e-6)
        x_norm = (x / s_x).clamp(-1, 1)
        
        # Flatten for MBE encoding
        x_flat = x_norm.view(-1, self.nx) 
        
        self.mbe.eval()
        with torch.no_grad():
            _, s_seq, d_seq_scaled = self.mbe(x_flat, return_sequences=True)
            
        T, N, B, I = s_seq.shape
        flat_contributions = (s_seq * d_seq_scaled).reshape(T*N, B, I)
        encoded_input_norm = flat_contributions.sum(dim=0)
        
        # Reshape to match x for scaling
        encoded_input_norm = encoded_input_norm.view(x.shape)
        encoded_input = encoded_input_norm * s_x
        
        # Conv1D logic
        size_out = x.size()[:-1] + (self.nf,)
        out = torch.addmm(self.bias, encoded_input.view(-1, self.nx), self.weight)
        return out.view(*size_out)

    def load_from_standard_conv1d(self, conv1d):
        with torch.no_grad():
            self.weight.copy_(conv1d.weight)
            self.bias.copy_(conv1d.bias)
