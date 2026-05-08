import torch
import torch.nn as nn
import math

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Fast sigmoid surrogate gradient for better gradient flow
        # gamma is a steepness parameter, e.g., 2.0
        gamma = 2.0
        surrogate_grad = gamma / (1.0 + gamma * torch.abs(x)) ** 2
        return grad_output * surrogate_grad

class MBENeuron(nn.Module):
    """
    Multi-basis Exponential Decay (MBE) Neuron — Fully Learnable Parametric Design

    Each basis (d, r, vth) now follows a (Scale * exp(-t/tau) + Bias) structure.
    This allows the neuron to adapt its firing rate and contribution intensity
    independently for both positive and negative input ranges.

    Formula for each parameter P ∈ {d, r, vth}:
        P_eff(t) = alpha_p · exp(-t·Δt / τ_p) + bias_p

    Key Advantage:
        By allowing bias_r to be learnable (and potentially negative), the neuron
        can avoid "deep reset" in negative input ranges, enabling dense spiking
        and higher approximation resolution for functions like Tanh.

    Learnable params: 
        - tau_{d,r,vth}, alpha_{d,r,vth}, bias_{d,r,vth}, w
    """
    def __init__(self, num_basis=4, timesteps=16, dt=1.0, alpha=1.0):
        super(MBENeuron, self).__init__()
        self.num_basis = num_basis
        self.timesteps = timesteps
        self.dt = dt

        # Decay time-constants (tau) — learnable.
        self.tau_d   = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)
        self.tau_r   = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)
        self.tau_vth = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)

        # 1. Scale (alpha) parameters: Conservative start (0.1 * alpha)
        half = num_basis // 2
        # Use 0.1 * alpha to avoid initial saturation, especially for functions with small values
        base_scale = 0.1 * float(alpha)
        init_alpha = torch.ones(num_basis) * base_scale
        
        # d (intensity) starts positive
        self.alpha_d = nn.Parameter(init_alpha.clone())
        
        # r (reset) and vth (threshold): split signs for positive/negative coverage
        init_alpha_r = init_alpha.clone()
        init_alpha_r[half:] *= -1.0
        self.alpha_r = nn.Parameter(init_alpha_r)
        
        init_alpha_vth = init_alpha.clone()
        init_alpha_vth[half:] *= -1.0
        self.alpha_vth = nn.Parameter(init_alpha_vth)

        # 2. Shift (bias) parameters
        self.bias_d = nn.Parameter(torch.zeros(num_basis))
        self.bias_r = nn.Parameter(torch.zeros(num_basis))
        
        # vth_bias: Positive group=0, Negative group=Random(-1 to 0) for softer start
        init_bias_vth = torch.zeros(num_basis)
        # Random offset between -1 and 0 for negative bases
        init_bias_vth[half:] = -torch.rand(num_basis - half)
        self.vth_bias = nn.Parameter(init_bias_vth)

        # 3. Basis combination weights w^(n): Standard random (full freedom)
        self.w = nn.Parameter(torch.randn(num_basis) / math.sqrt(num_basis))

        self.heaviside = SurrogateHeaviside.apply

    def forward(self, x, return_sequences=False):
        """
        Args:
            x: Input tensor of any shape.
            return_sequences: If True, returns (out, s_seq, d_seq_scaled)
                              for use by MBEMultiplier.
        Returns:
            Approximated output of the same shape as x (or a 3-tuple).
        """
        device = x.device
        dtype = x.dtype

        # All bases share the same initial membrane potential u[0] = x.
        u = x.unsqueeze(0).expand(self.num_basis, *x.shape).clone()
        o = torch.zeros_like(u)
        
        s_seq = [] if return_sequences else None
        
        # Time array for t=0 to T-1
        t_seq = torch.arange(self.timesteps, device=device, dtype=dtype) * self.dt
        
        # View shape for broadcasting sequence over batch dimensions: (T, N, 1, 1, ...)
        view_shape = (self.timesteps, self.num_basis) + (1,) * x.dim()
        
        # Clamp tau to avoid division by zero or negative tau during training
        tau_d = torch.clamp(self.tau_d, min=1e-3).to(device)
        tau_r = torch.clamp(self.tau_r, min=1e-3).to(device)
        tau_vth = torch.clamp(self.tau_vth, min=1e-3).to(device)
        
        # Precompute dynamically decaying parameters for all timesteps.
        alpha_d = self.alpha_d.to(device=device, dtype=dtype)
        alpha_r = self.alpha_r.to(device=device, dtype=dtype)
        alpha_vth = self.alpha_vth.to(device=device, dtype=dtype)
        
        # Expand biases for broadcasting: (num_basis, ...) -> (1, num_basis, ...)
        b_d = self.bias_d.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        b_r = self.bias_r.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        b_vth = self.vth_bias.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        
        # d_eff(t) = alpha_d * exp(-t/tau_d) + bias_d
        decay_d = alpha_d.view(1, self.num_basis, *([1]*x.dim())) * \
                  torch.exp(-t_seq.unsqueeze(1) / tau_d.unsqueeze(0)).view(view_shape) + b_d
                  
        # r_eff(t) = alpha_r * exp(-t/tau_r) + bias_r
        decay_r = alpha_r.view(1, self.num_basis, *([1]*x.dim())) * \
                  torch.exp(-t_seq.unsqueeze(1) / tau_r.unsqueeze(0)).view(view_shape) + b_r
        
        # vth_eff(t) = alpha_vth * exp(-t/tau_vth) + bias_vth
        vth_seq = alpha_vth.view(1, self.num_basis, *([1]*x.dim())) * \
                  torch.exp(-t_seq.unsqueeze(1) / tau_vth.unsqueeze(0)).view(view_shape) + b_vth

        # Iterate over timesteps
        for t in range(self.timesteps):
            d_t = decay_d[t]
            r_t = decay_r[t]
            vth_t = vth_seq[t]
            
            # Spike generation: sn[t] = H(un[t] - Vthn[t])
            s_t = self.heaviside(u - vth_t)
            
            if return_sequences:
                s_seq.append(s_t)
            
            # Accumulate weighted spike output; reset membrane potential
            o = o + s_t * d_t
            u = u - s_t * r_t
            
        # f(x) = Σ w^(n) · o^(n)  (paper eq. 8)
        # w^(n) is sign-free; bases with vth_bias < 0 fire on negative inputs,
        # and their negative w^(n) produces the correct signed output.
        w_view = self.w.view(self.num_basis, *([1] * x.dim())).to(device)
        out = torch.sum(w_view * o, dim=0)
        
        if return_sequences:
            # s_seq shape: (T, N, batch_size, ...)
            s_seq_tensor = torch.stack(s_seq, dim=0)
            
            # d_seq shape: (T, N, 1, ...)
            d_seq = decay_d
            
            # Scale d_seq by w: d_scaled shape: (T, N, 1, ...)
            d_seq_scaled = d_seq * w_view.unsqueeze(0)
            
            return out, s_seq_tensor, d_seq_scaled
            
        return out
        
    def save(self, filepath):
        """Saves the MBE Neuron parameters to a file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save({
            'num_basis': self.num_basis,
            'timesteps': self.timesteps,
            'dt': self.dt,
            'state_dict': self.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath, device='cpu'):
        """Loads an MBE Neuron from a file."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            num_basis=checkpoint['num_basis'], 
            timesteps=checkpoint['timesteps'], 
            dt=checkpoint['dt']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

def train_mbe_neuron(target_func, x_range=(-10, 10), num_samples=10000, num_epochs=5000, lr=0.01, tv_weight=0.0, device=None, target_loss=1e-4, **mbe_kwargs):
    """
    Utility function to optimize MBE Neuron parameters to fit a specific target function.
    
    Args:
        target_func: The function to approximate (e.g., torch.nn.functional.gelu)
        x_range: Tuple (min, max) for sampling input values
        num_samples: Number of samples M
        num_epochs: Training epochs
        lr: Learning rate
        tv_weight: Weight for Total Variation regularization to reduce oscillations
        device: Device to use (cpu or cuda)
        target_loss: Early stopping threshold for MSE loss
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mbe = MBENeuron(**mbe_kwargs).to(device)
    optimizer = torch.optim.Adam(mbe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    criterion = nn.MSELoss()
    
    # Generate training data
    x = torch.linspace(x_range[0], x_range[1], num_samples, device=device).unsqueeze(1)
    y_target = target_func(x)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = mbe(x)
        loss = criterion(y_pred, y_target)
        
        # Total Variation Regularization (optional)
        if tv_weight > 0:
            tv_loss = torch.mean(torch.abs(y_pred[1:] - y_pred[:-1]))
            loss += tv_weight * tv_loss
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
            
        # Early Stopping
        if loss.item() < target_loss:
            print(f"Early stopping at epoch {epoch+1} with loss {loss.item():.6f}")
            break
            
    return mbe
