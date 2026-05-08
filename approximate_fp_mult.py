import torch
import torch.nn as nn

class MBEMultiplier(nn.Module):
    """
    MBE-based FP Multiplier using Spike-Intensity Interaction. (True Hardware Simulation)
    Approximates x1 * x2 = sum_i,j (s1_i * s2_j * d1_i * d2_j)
    
    This version explicitly constructs the (K, K) interaction matrices for every element 
    in the batch, exactly as it would happen in a neuromorphic crossbar array.
    """
    def __init__(self, mbe_id_model=None, num_basis=4, timesteps=16, dt=1.0, alpha=1.0):
        super(MBEMultiplier, self).__init__()
        from MBE_neurons import MBENeuron
        if mbe_id_model is None:
            self.mbe_id = MBENeuron(num_basis=num_basis, timesteps=timesteps, dt=dt, alpha=alpha)
        else:
            self.mbe_id = mbe_id_model

    def forward(self, x1, x2):
        """
        x1, x2: Tensors of same shape (Batch, ...)
        Returns: Approximated product tensor
        """
        # 1. Encode both operands into spike sequences: (T, N, Batch, Features)
        _, s1, d1 = self.mbe_id(x1, return_sequences=True)
        _, s2, d2 = self.mbe_id(x2, return_sequences=True)
        
        orig_shape = x1.shape
        T, N = s1.shape[0], s1.shape[1]
        K = T * N
        
        # Flatten time and basis: (K, Total_Elements)
        # s1_flat, s2_flat: (K, B*F)
        s1_flat = s1.view(K, -1)
        s2_flat = s2.view(K, -1)
        d1_flat = d1.view(K, -1)
        d2_flat = d2.view(K, -1)
        
        total_elements = s1_flat.shape[1]
        
        # 2. Hardware-style Matrix Interaction (The "Standard Way")
        # We need to compute S_ij = s1_i * s2_j and D_ij = d1_i * d2_j
        
        # Reshape for batch-wise outer product
        # (Total_Elements, K, 1) and (Total_Elements, 1, K)
        s1_b = s1_flat.t().unsqueeze(2) 
        s2_b = s2_flat.t().unsqueeze(1)
        
        d1_b = d1_flat.t().unsqueeze(2)
        d2_b = d2_flat.t().unsqueeze(1)
        
        # Interaction Matrices (Batch, K, K)
        S = torch.matmul(s1_b, s2_b) # Spike interactions (Binary 0/1)
        D = torch.matmul(d1_b, d2_b) # Intensity interactions
        
        # Track SOPs for the paper's methodology
        # Each '1' in S represents a spiking interaction (SOP)
        self.last_interaction_sops = S.sum().item()
        
        # 3. Final Sum of Interactions: y = sum(S * D)
        approx_product_flat = torch.sum(S * D, dim=(1, 2))
        
        # Reshape back to (Batch, Features)
        return approx_product_flat.view(orig_shape)
