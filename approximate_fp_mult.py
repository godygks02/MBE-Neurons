import torch
import torch.nn as nn

class MBEMultiplier(nn.Module):
    """
    Optimized MBE-based FP Multiplier.
    Avoids explicit (K, K) matrix construction to save memory and time.
    y = sum_i(s1_i * d1_i) * sum_j(s2_j * d2_j)
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
        Efficiently approximates x1 * x2 using the distributive property.
        Instead of a (K, K) interaction matrix, we sum the decoded contributions 
        of each operand independently and then multiply.
        """
        # 1. Encode both operands and get decoded values directly
        # MBE forward returns (decoded_val, s_seq, d_seq_scaled)
        # We only need the decoded_val for the multiplication logic, 
        # but we use the spiking framework to maintain SNN properties.
        
        # decoded1 = sum_t,n (s1_tn * d1_tn)
        decoded1, _, _ = self.mbe_id(x1, return_sequences=True)
        # decoded2 = sum_t,n (s2_tn * d2_tn)
        decoded2, _, _ = self.mbe_id(x2, return_sequences=True)
        
        # The theoretical MBE product is exactly the product of the decoded values
        # because (sum s1 d1) * (sum s2 d2) = sum_i,j (s1_i s2_j d1_i d2_j)
        return decoded1 * decoded2
