import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SurrogateTernarySpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, vth_pos, vth_neg):
        ctx.save_for_backward(u, vth_pos, vth_neg)
        s = torch.zeros_like(u)
        s[u > vth_pos] = 1.0
        s[u < vth_neg] = -1.0
        return s

    @staticmethod
    def backward(ctx, grad_output):
        u, vth_pos, vth_neg = ctx.saved_tensors
        gamma = 2.0
        
        # Dual Surrogate Gradient
        sg_pos = gamma / (1.0 + gamma * torch.abs(u - vth_pos)) ** 2
        sg_neg = gamma / (1.0 + gamma * torch.abs(u - vth_neg)) ** 2
        
        grad_u = grad_output * (sg_pos + sg_neg)
        grad_vth_pos = grad_output * (-sg_pos)
        grad_vth_neg = grad_output * (-sg_neg)
        
        return grad_u, grad_vth_pos, grad_vth_neg

class TernaryEAMBENeuron(nn.Module):
    """
    Ternary Extended-Adaptive MBE (EA-MBE) Neuron
    
    Combines Extended-Adaptive (EA) parameters with Ternary Spikes {-1, 0, 1}.
    This neuron can fire positive spikes when u > Vth, and negative spikes when u < -Vth.
    It solves the 'Dead Neuron' problem by directly expressing negative values 
    through negative spikes and utilizing a dual surrogate gradient.
    """
    def __init__(self, num_basis=4, timesteps=16, dt=1.0, alpha=1.0):
        super(TernaryEAMBENeuron, self).__init__()
        self.num_basis = num_basis
        self.timesteps = timesteps
        self.dt = dt

        # Decay time-constants (tau) — learnable
        self.tau_d   = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)
        self.tau_r   = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)
        self.tau_vth_pos = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)
        self.tau_vth_neg = nn.Parameter(torch.rand(num_basis) * 5.0 + 1.0)

        # 1. Scale (alpha) parameters: Asymmetric start to prevent rank collapse
        base_scale = 0.1 * float(alpha)
        
        # We add small random noise to alpha to break symmetry
        self.alpha_d = nn.Parameter(torch.ones(num_basis) * base_scale + torch.randn(num_basis) * base_scale * 0.1)
        self.alpha_r = nn.Parameter(torch.ones(num_basis) * base_scale + torch.randn(num_basis) * base_scale * 0.1)
        self.alpha_vth_pos = nn.Parameter(torch.ones(num_basis) * base_scale + torch.randn(num_basis) * base_scale * 0.1)
        self.alpha_vth_neg = nn.Parameter(torch.ones(num_basis) * base_scale + torch.randn(num_basis) * base_scale * 0.1)

        # 2. Shift (bias / beta) parameters - The "Extended-Adaptive" core
        # Asymmetric bias initialization
        self.bias_d = nn.Parameter(torch.randn(num_basis))
        self.bias_r = nn.Parameter(torch.randn(num_basis))
        self.bias_vth_pos = nn.Parameter(torch.randn(num_basis))
        self.bias_vth_neg = nn.Parameter(torch.randn(num_basis))

        # 3. Basis combination weights w^(n)
        self.w = nn.Parameter(torch.randn(num_basis) / math.sqrt(num_basis))

        self.ternary_spike = SurrogateTernarySpike.apply

    def forward(self, x, return_sequences=False):
        """
        Args:
            x: Input tensor of any shape.
            return_sequences: If True, returns (out, s_seq, d_seq_scaled)
                              for use by MBEMultiplier or analysis.
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
        tau_vth_pos = torch.clamp(self.tau_vth_pos, min=1e-3).to(device)
        tau_vth_neg = torch.clamp(self.tau_vth_neg, min=1e-3).to(device)
        
        # Precompute dynamically decaying parameters for all timesteps.
        alpha_d = self.alpha_d.to(device=device, dtype=dtype)
        alpha_r = self.alpha_r.to(device=device, dtype=dtype)
        alpha_vth_pos = self.alpha_vth_pos.to(device=device, dtype=dtype)
        alpha_vth_neg = self.alpha_vth_neg.to(device=device, dtype=dtype)
        
        # Expand biases for broadcasting
        b_d = self.bias_d.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        b_r = self.bias_r.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        b_vth_pos = self.bias_vth_pos.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        b_vth_neg = self.bias_vth_neg.view(1, self.num_basis, *([1]*x.dim())).to(device=device, dtype=dtype)
        
        # d_eff(t) = alpha_d * exp(-t/tau_d) + beta_d
        decay_d = alpha_d.view(1, self.num_basis, *([1]*x.dim())) * \
                  torch.exp(-t_seq.unsqueeze(1) / tau_d.unsqueeze(0)).view(view_shape) + b_d
                  
        # r_eff(t) = alpha_r * exp(-t/tau_r) + beta_r
        decay_r = alpha_r.view(1, self.num_basis, *([1]*x.dim())) * \
                  torch.exp(-t_seq.unsqueeze(1) / tau_r.unsqueeze(0)).view(view_shape) + b_r
        
        # vth_pos_seq(t)
        vth_pos_decay = F.softplus(alpha_vth_pos.view(1, self.num_basis, *([1]*x.dim()))) * \
                        torch.exp(-t_seq.unsqueeze(1) / tau_vth_pos.unsqueeze(0)).view(view_shape)
        vth_pos_seq = vth_pos_decay + b_vth_pos

        # vth_neg_seq(t)
        vth_neg_decay = F.softplus(alpha_vth_neg.view(1, self.num_basis, *([1]*x.dim()))) * \
                        torch.exp(-t_seq.unsqueeze(1) / tau_vth_neg.unsqueeze(0)).view(view_shape)
        vth_neg_seq = -vth_neg_decay + b_vth_neg

        # Iterate over timesteps
        for t in range(self.timesteps):
            d_t = decay_d[t]
            r_t = decay_r[t]
            
            vth_pos = vth_pos_seq[t]
            vth_neg = vth_neg_seq[t]
            
            # Ternary Spike generation: s_t in {-1, 0, 1}
            s_t = self.ternary_spike(u, vth_pos, vth_neg)
            
            if return_sequences:
                s_seq.append(s_t)
            
            # Accumulate weighted spike output; reset membrane potential
            # Since s_t can be -1, o subtracts d_t and u adds r_t naturally.
            o = o + s_t * d_t
            u = u - s_t * r_t
            
        # f(x) = Σ w^(n) · o^(n)
        w_view = self.w.view(self.num_basis, *([1] * x.dim())).to(device)
        out = torch.sum(w_view * o, dim=0)
        
        if return_sequences:
            s_seq_tensor = torch.stack(s_seq, dim=0)
            d_seq_scaled = decay_d * w_view.unsqueeze(0)
            return out, s_seq_tensor, d_seq_scaled
            
        return out
        
    def save(self, filepath):
        """Saves the Ternary EA-MBE Neuron parameters to a file."""
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
        """Loads a Ternary EA-MBE Neuron from a file."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            num_basis=checkpoint['num_basis'], 
            timesteps=checkpoint['timesteps'], 
            dt=checkpoint['dt']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model

def train_ternary_mbe_neuron(target_func, x_range=(-10, 10), num_samples=10000, num_epochs=5000, lr=0.01, tv_weight=0.0, l1_spike_weight=0.0, device=None, target_loss=1e-4, patience=500, **mbe_kwargs):
    """
    Utility function to optimize Ternary EA-MBE Neuron parameters to fit a target function.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mbe = TernaryEAMBENeuron(**mbe_kwargs).to(device)
    optimizer = torch.optim.Adam(mbe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr*0.01)
    criterion = nn.MSELoss()
    
    # Generate training data
    x = torch.linspace(x_range[0], x_range[1], num_samples, device=device).unsqueeze(1)
    y_target = target_func(x)
    
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        if l1_spike_weight > 0:
            y_pred, s_seq, _ = mbe(x, return_sequences=True)
        else:
            y_pred = mbe(x)
            
        loss = criterion(y_pred, y_target)
        
        # L1 Spike Regularization for Ternary spikes (using absolute value)
        if l1_spike_weight > 0:
            spike_loss = torch.abs(s_seq).mean()
            loss += l1_spike_weight * spike_loss
        
        # Total Variation Regularization (optional)
        if tv_weight > 0:
            tv_loss = torch.mean(torch.abs(y_pred[1:] - y_pred[:-1]))
            loss += tv_weight * tv_loss
            
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_val = loss.item()
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_val:.6f}")
            
        # Early Stopping on Target
        if loss_val < target_loss:
            print(f"Early stopping at epoch {epoch+1} with loss {loss_val:.6f} < target {target_loss}")
            break
            
        # Early Stopping on Oscillation/Stagnation
        if loss_val < best_loss - 1e-7:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to oscillation/stagnation. Best loss: {best_loss:.6f}")
            break
            
    return mbe
