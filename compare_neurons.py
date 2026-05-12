import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import random
import numpy as np
from MBE_neurons import MBENeuron, train_mbe_neuron
from Ternary_EA_MBE_neurons import TernaryEAMBENeuron, train_ternary_mbe_neuron

def set_seed(seed):
    """Fix seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_target_function(name):
    functions = {
        'gelu': F.gelu,
        'relu': F.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'silu': F.silu,
    }
    
    name = name.lower()
    if name not in functions:
        raise ValueError(f"Target function '{name}' is not supported. Supported functions: {list(functions.keys())}")
    
    return functions[name]

def main():
    parser = argparse.ArgumentParser(description="Compare Standard MBE Neuron vs Ternary EA-MBE Neuron")
    parser.add_argument('--target', type=str, default='gelu', help='Target activation function (e.g., gelu, relu, sigmoid, tanh, silu)')
    parser.add_argument('--num_basis', type=int, default=4, help='Number of basis components (N)')
    parser.add_argument('--timesteps', type=int, default=16, help='Number of timesteps (T)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--x_min', type=float, default=-10.0, help='Minimum x value for sampling')
    parser.add_argument('--x_max', type=float, default=10.0, help='Maximum x value for sampling')
    parser.add_argument('--tv_weight', type=float, default=0.0, help='Weight for Total Variation regularization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true', help='Plot the approximation result')
    
    args = parser.parse_args()
    
    print(f"=== Comparing {args.target.upper()} with MBE vs Ternary EA-MBE ===")
    print(f"Config: Basis={args.num_basis}, Timesteps={args.timesteps}, Range=({args.x_min}, {args.x_max})")
    
    target_func = get_target_function(args.target)
    
    temp_x = torch.linspace(args.x_min, args.x_max, 1000)
    target_y = target_func(temp_x)
    estimated_alpha = float(target_y.max().item())
    print(f"Estimated Alpha (Max value): {estimated_alpha:.4f}")
    
    if estimated_alpha < 1e-5:
        estimated_alpha = 1.0
        
    print("\n--- Training Standard MBE Neuron ---")
    set_seed(args.seed)
    mbe_model = train_mbe_neuron(
        target_func=target_func, 
        x_range=(args.x_min, args.x_max), 
        num_samples=10000, 
        num_epochs=args.epochs, 
        lr=args.lr, 
        tv_weight=args.tv_weight,
        num_basis=args.num_basis, 
        timesteps=args.timesteps, 
        alpha=estimated_alpha
    )
    
    print("\n--- Training Ternary EA-MBE Neuron ---")
    set_seed(args.seed)
    ternary_model = train_ternary_mbe_neuron(
        target_func=target_func, 
        x_range=(args.x_min, args.x_max), 
        num_samples=10000, 
        num_epochs=args.epochs, 
        lr=args.lr, 
        tv_weight=args.tv_weight,
        num_basis=args.num_basis, 
        timesteps=args.timesteps, 
        alpha=estimated_alpha
    )
    
    device = next(mbe_model.parameters()).device
    x_test = torch.linspace(args.x_min, args.x_max, 1000, device=device).unsqueeze(1)
    
    mbe_model.eval()
    ternary_model.eval()
    with torch.no_grad():
        y_true = target_func(x_test)
        y_pred_mbe = mbe_model(x_test)
        y_pred_ternary = ternary_model(x_test)
        
    mse_mbe = F.mse_loss(y_pred_mbe, y_true).item()
    mse_ternary = F.mse_loss(y_pred_ternary, y_true).item()
    
    print(f"\n=== Final Comparison Results ({args.target.upper()}) ===")
    print(f"Standard MBE MSE:   {mse_mbe:.6f}")
    print(f"Ternary EA-MBE MSE: {mse_ternary:.6f}")
    print(f"Improvement Factor: {(mse_mbe / (mse_ternary + 1e-10)):.2f}x")
    
    if args.plot:
        x_cpu = x_test.cpu().numpy()
        y_true_cpu = y_true.cpu().numpy()
        y_mbe_cpu = y_pred_mbe.cpu().numpy()
        y_ternary_cpu = y_pred_ternary.cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Main Plot
        ax1.plot(x_cpu, y_true_cpu, label=f"Target: {args.target.upper()}", linewidth=6, color='black', alpha=0.2)
        ax1.plot(x_cpu, y_mbe_cpu, label=f"Standard MBE (MSE: {mse_mbe:.5f})", linestyle='--', linewidth=2.5, color='blue', alpha=0.8)
        ax1.plot(x_cpu, y_ternary_cpu, label=f"Ternary EA-MBE (MSE: {mse_ternary:.5f})", linestyle=':', linewidth=3.0, color='red')
        ax1.set_title(f"Comparison: MBE vs Ternary EA-MBE Neuron on {args.target.upper()}")
        ax1.set_ylabel("Output f(x)")
        ax1.legend()
        ax1.grid(True)
        
        # Error (Residuals) Plot
        error_mbe = np.abs(y_mbe_cpu - y_true_cpu)
        error_ternary = np.abs(y_ternary_cpu - y_true_cpu)
        
        ax2.plot(x_cpu, error_mbe, label="Standard MBE Error", color='blue', alpha=0.7)
        ax2.plot(x_cpu, error_ternary, label="Ternary EA-MBE Error", color='red', alpha=0.7)
        ax2.set_xlabel("Input x")
        ax2.set_ylabel("Absolute Error")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        save_dir = os.path.join('plots', 'comparison')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"compare_{args.target}_approximation.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
