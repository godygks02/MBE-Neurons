import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import os
import random
import numpy as np
from MBE_neurons import MBENeuron, train_mbe_neuron

def set_seed(seed):
    """Fix seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_target_function(name):
    """
    Returns the target activation function based on the name.
    Easily extendable to support more functions.
    """
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
    parser = argparse.ArgumentParser(description="Approximate Activation Functions using MBE Neuron")
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
    
    set_seed(args.seed)
    
    print(f"=== Approximating {args.target.upper()} with MBE Neuron ===")
    print(f"Config: Basis={args.num_basis}, Timesteps={args.timesteps}, Range=({args.x_min}, {args.x_max})")
    
    target_func = get_target_function(args.target)
    
    # 1. Estimate alpha (max value of target function in the range)
    temp_x = torch.linspace(args.x_min, args.x_max, 1000)
    target_y = target_func(temp_x)
    estimated_alpha = float(target_y.max().item())
    print(f"Estimated Alpha (Max value): {estimated_alpha:.4f}")
    
    # If alpha is 0 or very small (e.g. constant 0), default to 1.0 to avoid numerical issues
    if estimated_alpha < 1e-5:
        estimated_alpha = 1.0
        
    import os
    model_dir = os.path.join('mbe_models', 'activation')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"mbe_{args.target}_model.pth")
    
    if os.path.exists(model_path):
        print(f"\nLoading pre-trained MBE Neuron from {model_path} ...")
        trained_mbe = MBENeuron.load(model_path)
    else:
        print("\nStarting Training...")
        trained_mbe = train_mbe_neuron(
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
        trained_mbe.save(model_path)
        print("\nTraining Completed.")
    
    # 3. Evaluate and Visualize
    device = next(trained_mbe.parameters()).device
    x_test = torch.linspace(args.x_min, args.x_max, 1000, device=device).unsqueeze(1)
    
    trained_mbe.eval()
    with torch.no_grad():
        y_pred = trained_mbe(x_test)
        y_true = target_func(x_test)
        
    mse_loss = F.mse_loss(y_pred, y_true).item()
    print(f"Final MSE Loss on evaluation: {mse_loss:.6f}")
    
    if args.plot:
        x_cpu = x_test.cpu().numpy()
        y_true_cpu = y_true.cpu().numpy()
        y_pred_cpu = y_pred.cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_cpu, y_true_cpu, label=f"Target: {args.target.upper()}", linewidth=2, color='blue')
        plt.plot(x_cpu, y_pred_cpu, label=f"MBE Approximation", linestyle='--', linewidth=2, color='red')
        plt.title(f"MBE Neuron Approximation for {args.target.upper()}\nMSE: {mse_loss:.6f}")
        plt.xlabel("Input x")
        plt.ylabel("Output f(x)")
        plt.legend()
        plt.grid(True)
        
        save_dir = os.path.join('plots', 'activation')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"mbe_{args.target}_approximation.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()
