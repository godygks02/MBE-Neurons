import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt

from data_utils import generate_data
from model_utils import ToyTransformerMLP, evaluate_with_metrics, get_device, calculate_ann_energy
from mbe_modules import MBEGELU, MBELinear, MBELayerNorm, MBESoftmax
from MBE_neurons import MBENeuron, train_mbe_neuron

# SNN Model Wrapper
class ToyTransformerSNN(nn.Module):
    def __init__(self, ann_model, timesteps=16, num_basis=8):
        super().__init__()
        self.fc1 = MBELinear(ann_model.fc1.in_features, ann_model.fc1.out_features, timesteps=timesteps, num_basis=num_basis)
        self.ln1 = MBELayerNorm(ann_model.ln1.normalized_shape[0], timesteps=timesteps, num_basis=num_basis)
        self.gelu1 = MBEGELU(timesteps=timesteps, num_basis=num_basis)
        
        self.fc2 = MBELinear(ann_model.fc2.in_features, ann_model.fc2.out_features, timesteps=timesteps, num_basis=num_basis)
        self.ln2 = MBELayerNorm(ann_model.ln2.normalized_shape[0], timesteps=timesteps, num_basis=num_basis)
        self.gelu2 = MBEGELU(timesteps=timesteps, num_basis=num_basis)
        
        self.fc3 = MBELinear(ann_model.fc3.in_features, ann_model.fc3.out_features, timesteps=timesteps, num_basis=num_basis)
        self.softmax = MBESoftmax(timesteps=timesteps, num_basis=num_basis)

        self.fc1.load_from_standard_linear(ann_model.fc1)
        self.ln1.load_from_standard_layernorm(ann_model.ln1)
        self.fc2.load_from_standard_linear(ann_model.fc2)
        self.ln2.load_from_standard_layernorm(ann_model.ln2)
        self.fc3.load_from_standard_linear(ann_model.fc3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu2(x)
        logits = self.fc3(x)
        return logits

def calibrate_ranges(model, loader, device):
    model.eval()
    ranges = {'fc1_in': [], 'fc1_w': [], 'ln1_in': [], 'gelu1_in': [], 
              'fc2_in': [], 'fc2_w': [], 'ln2_in': [], 'gelu2_in': [], 
              'fc3_in': [], 'fc3_w': []}
    
    ranges['fc1_w'] = (model.fc1.weight.min().item(), model.fc1.weight.max().item())
    ranges['fc2_w'] = (model.fc2.weight.min().item(), model.fc2.weight.max().item())
    ranges['fc3_w'] = (model.fc3.weight.min().item(), model.fc3.weight.max().item())

    def hook_fn(name):
        def hook(module, input, output):
            ranges[name].append((input[0].min().item(), input[0].max().item()))
        return hook

    hooks = [
        model.fc1.register_forward_hook(hook_fn('fc1_in')),
        model.ln1.register_forward_hook(hook_fn('ln1_in')),
        model.gelu1.register_forward_hook(hook_fn('gelu1_in')),
        model.fc2.register_forward_hook(hook_fn('fc2_in')),
        model.ln2.register_forward_hook(hook_fn('ln2_in')),
        model.gelu2.register_forward_hook(hook_fn('gelu2_in')),
        model.fc3.register_forward_hook(hook_fn('fc3_in')),
    ]

    with torch.no_grad():
        for batch_x, _ in loader:
            model(batch_x.to(device))
    for h in hooks: h.remove()
    
    final_ranges = {}
    for k, v in ranges.items():
        if isinstance(v, tuple):
            final_ranges[k] = (round(v[0], 2), round(v[1], 2))
        else:
            final_ranges[k] = (round(min([x[0] for x in v]), 2), round(max([x[1] for x in v]), 2))
    return final_ranges

def main():
    parser = argparse.ArgumentParser(description="Test SNN converted from ANN with MBE neurons")
    parser.add_argument('--load_name', type=str, default='mnist_mlp.pth', help='Model filename to load')
    parser.add_argument('--timesteps', type=int, default=16, help='Number of timesteps (T)')
    parser.add_argument('--num_basis', type=int, default=4, help='Number of basis functions (N)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--force_train', action='store_true', help='Force fresh training of MBE neurons')
    
    args = parser.parse_args()
    device = get_device()
    
    # 1. Fixed Paths
    exp_dir = "mbe_models/mlp2snn"
    plot_dir = "plots/mlp2snn"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # 2. Load trained ANN with metadata
    ann_path = os.path.join(exp_dir, args.load_name)
    if not os.path.exists(ann_path):
        print(f"Error: Trained model not found at {ann_path}. Run train_ann.py first.")
        return
    
    print(f"Loading ANN from {ann_path}...")
    checkpoint = torch.load(ann_path, map_location=device)
    
    ann_model = ToyTransformerMLP(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes']
    ).to(device)
    ann_model.load_state_dict(checkpoint['state_dict'])
    
    # Load MNIST data
    train_loader, test_loader = generate_data(batch_size=args.batch_size)
    
    ann_acc, _, _ = evaluate_with_metrics(ann_model, test_loader, device)
    print(f"Loaded ANN Test Accuracy: {ann_acc:.2f}%")

    # 3. Calibration & Conversion
    print("\nInvestigating activation ranges...")
    final_ranges = calibrate_ranges(ann_model, train_loader, device)
    
    print(f"\nConverting to SNN (T={args.timesteps}, N={args.num_basis}) and fitting MBE neurons...")
    snn_model = ToyTransformerSNN(ann_model, timesteps=args.timesteps, num_basis=args.num_basis).to(device)
    
    # Use range-specific names to allow reuse across different experiments with same distribution
    def get_mbe_name(base, r, t, n):
        return f"{base}_R({r[0]:.2f}_{r[1]:.2f})_T{t}_N{n}.pth"

    snn_model.gelu1.model_path = os.path.join(exp_dir, get_mbe_name("gelu1", final_ranges['gelu1_in'], args.timesteps, args.num_basis))
    snn_model.gelu1.fit(x_range=final_ranges['gelu1_in'], epochs=5000)
    
    snn_model.gelu2.model_path = os.path.join(exp_dir, get_mbe_name("gelu2", final_ranges['gelu2_in'], args.timesteps, args.num_basis))
    snn_model.gelu2.fit(x_range=final_ranges['gelu2_in'], epochs=5000)
    
    # Determine ID range and name
    id_range = (-1.0, 1.0)
    id_mbe_name = get_mbe_name("id_mbe", id_range, args.timesteps, args.num_basis)
    id_mbe_path = os.path.join(exp_dir, id_mbe_name)
    if os.path.exists(id_mbe_path) and not args.force_train:
        mbe_id = MBENeuron.load(id_mbe_path, device=device)
    else:
        mbe_id = train_mbe_neuron(lambda x: x, x_range=(-1.0, 1.0), num_basis=args.num_basis, timesteps=args.timesteps, device=device, alpha=1.0, num_epochs=5000, target_loss=1e-5)
        mbe_id.save(id_mbe_path)
    
    for fc in [snn_model.fc1, snn_model.fc2, snn_model.fc3]:
        fc.initialize_multiplier(mbe_id)

    # 4. Verification
    # 4. Verification
    print("\nEvaluating SNN Metrics (Paper Methodology)...")
    snn_acc, avg_sops, avg_eta = evaluate_with_metrics(snn_model, test_loader, device)
    
    # 5. CDCER (Comprehensive Dynamic Compute Efficiency Ratio) Analysis
    hidden_dim = checkpoint['hidden_dim']
    num_classes = checkpoint['num_classes']
    in_dim = checkpoint['input_dim']
    T = args.timesteps
    N = args.num_basis

    # --- 5.1 ANN Energy Baseline (High-Fidelity PDF Methodology) ---
    energy_ann = calculate_ann_energy(in_dim, hidden_dim, num_classes)
    total_ann_energy = energy_ann['total_pj']
    e_ann_linear = energy_ann['breakdown_pj']['linear']
    e_ann_ln = energy_ann['breakdown_pj']['ln']
    e_ann_gelu = energy_ann['breakdown_pj']['gelu']
    e_ann_softmax = energy_ann['breakdown_pj']['softmax']

    # --- 5.2 SNN Energy (Inference + Pre-compute) ---
    # SNN Inference Energy (Pure SOPs at 0.9 pJ)
    e_snn_inference = avg_sops * 0.9
    
    # SNN Pre-computation Energy (One-time cost to cache Weff = W * dn)
    # Only for Linear layers weights
    num_weights = (in_dim * hidden_dim) + (hidden_dim * hidden_dim) + (hidden_dim * num_classes)
    e_snn_precompute = (num_weights * N * T) * 3.7
    
    # Total SNN Energy (Amortized over K samples, K=10000 for evaluation)
    K = 10000 
    total_snn_energy = (e_snn_precompute / K) + e_snn_inference
    
    cdcer = (1 - total_snn_energy / total_ann_energy) * 100

    print(f"\n--- CDCER (Comprehensive Dynamic Compute Efficiency Ratio) ---")
    print(f"ANN Acc: {ann_acc:.2f}%, SNN Acc: {snn_acc:.2f}% (Delta: {ann_acc - snn_acc:.2f}%)")
    print(f"Avg Firing Rate (eta): {avg_eta*100:.2f}%")
    print(f"ANN Total Energy: {total_ann_energy/1e6:.2f} uJ (Non-linear overhead included)")
    print(f"SNN Total Energy: {total_snn_energy/1e6:.2f} uJ (Pre-compute amortized)")
    print(f"CDCER Score: {cdcer:.2f}%")
    
    # 6. Result Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Pure Inference Gain (excluding pre-compute)
    pure_gain = (1 - e_snn_inference / total_ann_energy) * 100

    table_data = [
        ["Metric", "ANN (Baseline)", f"SNN (T={T}, N={N})", "Efficiency Gain (CDCER)"],
        ["Accuracy", f"{ann_acc:.2f}%", f"{snn_acc:.2f}%", f"{ann_acc - snn_acc:.2f}% (Drop)"],
        ["Linear Energy", f"{e_ann_linear/1e6:.2f} uJ", f"Included in SOPs", "-"],
        ["Non-Linear Energy", f"{(e_ann_ln+e_ann_gelu+e_ann_softmax)/1e6:.2f} uJ", f"Included in SOPs", "-"],
        ["Pre-compute Cost", "-", f"{e_snn_precompute/1e6:.2f} uJ (Total)", "One-time Setup"],
        ["Total Dynamic Energy", f"{total_ann_energy/1e6:.2f} uJ", f"{total_snn_energy/1e6:.2f} uJ", f"{cdcer:.2f}%"],
        ["Pure Inference Gain", "-", "-", f"{pure_gain:.2f}% (Excl. Setup)"]
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.2, 0.25, 0.3, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')
            
    plt.title(f"Toy Transformer MLP Verification Report\n(Config: {args.load_name})", pad=20, weight='bold')
    plt.savefig(os.path.join(plot_dir, "result.png"), dpi=150, bbox_inches='tight')
    print(f"\nVerification Complete. Numerical report saved to {plot_dir}/result.png")

if __name__ == "__main__":
    main()
