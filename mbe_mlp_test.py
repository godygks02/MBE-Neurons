import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from mbe_modules import MBEGELU, MBELinear, MBELayerNorm, MBESoftmax
from MBE_neurons import MBENeuron, train_mbe_neuron

# 1. Config
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset Generation
def generate_data(n_samples=3000, n_features=20, centers=3, cluster_std=2.0):
    X_np, y_np = make_blobs(n_samples=n_samples, n_features=n_features, 
                            centers=centers, cluster_std=cluster_std, random_state=SEED)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader

# 3. ANN Model
class ToyTransformerMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gelu1 = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.gelu2 = nn.GELU()
        
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu2(x)
        logits = self.fc3(x)
        return logits

    def predict_probs(self, x):
        logits = self.forward(x)
        return self.softmax(logits)

# 4. SNN Model (MBE-based)
class ToyTransformerSNN(nn.Module):
    def __init__(self, ann_model, timesteps=16, num_basis=8):
        super().__init__()
        self.timesteps = timesteps
        self.num_basis = num_basis
        
        # Structure matches ANN
        self.fc1 = MBELinear(ann_model.fc1.in_features, ann_model.fc1.out_features, timesteps=timesteps, num_basis=num_basis)
        self.ln1 = MBELayerNorm(ann_model.ln1.normalized_shape[0], timesteps=timesteps, num_basis=num_basis)
        self.gelu1 = MBEGELU(timesteps=timesteps, num_basis=num_basis)
        
        self.fc2 = MBELinear(ann_model.fc2.in_features, ann_model.fc2.out_features, timesteps=timesteps, num_basis=num_basis)
        self.ln2 = MBELayerNorm(ann_model.ln2.normalized_shape[0], timesteps=timesteps, num_basis=num_basis)
        self.gelu2 = MBEGELU(timesteps=timesteps, num_basis=num_basis)
        
        self.fc3 = MBELinear(ann_model.fc3.in_features, ann_model.fc3.out_features, timesteps=timesteps, num_basis=num_basis)
        self.softmax = MBESoftmax(timesteps=timesteps, num_basis=num_basis)

        # Copy weights
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

    def predict_probs(self, x):
        logits = self.forward(x)
        return self.softmax(logits)

# 5. Training and Evaluation Functions
def train_ann(model, train_loader, epochs=10):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training ANN for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            acc, _ = evaluate_with_metrics(model, train_loader)
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%")

def evaluate_with_metrics(model, loader):
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    total_spikes = 0
    total_ops = 0 # MACs for ANN, SOPs for SNN
    
    # We will use a hook to count spikes in MBENeuron
    spike_counts = []
    def spike_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            # output is (out, s_seq, d_seq_scaled)
            s_seq = output[1] # (T, N, Batch, ...)
            spikes = s_seq.sum().item()
            spike_counts.append(spikes)

    hooks = []
    for m in model.modules():
        if isinstance(m, MBENeuron):
            hooks.append(m.register_forward_hook(spike_hook))

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            spike_counts = []
            outputs = model(batch_x)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            total_spikes += sum(spike_counts)

    for h in hooks: h.remove()
    acc = 100 * correct / total
    avg_spikes_per_sample = total_spikes / total
    
    return acc, avg_spikes_per_sample

# 6. Calibration
def calibrate_ranges(model, loader):
    model.eval()
    ranges = {
        'fc1_in': [], 'fc1_w': [],
        'ln1_in': [], 
        'gelu1_in': [],
        'fc2_in': [], 'fc2_w': [],
        'ln2_in': [],
        'gelu2_in': [],
        'fc3_in': [], 'fc3_w': [],
    }
    
    # Store weights ranges
    ranges['fc1_w'] = (model.fc1.weight.min().item(), model.fc1.weight.max().item())
    ranges['fc2_w'] = (model.fc2.weight.min().item(), model.fc2.weight.max().item())
    ranges['fc3_w'] = (model.fc3.weight.min().item(), model.fc3.weight.max().item())

    # Hooks to collect activation ranges
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
            model(batch_x.to(DEVICE))
            
    for h in hooks: h.remove()
    
    # Process ranges to get final (min, max)
    final_ranges = {}
    for key, val in ranges.items():
        if isinstance(val, tuple):
            # Round weights ranges
            final_ranges[key] = (round(val[0], 2), round(val[1], 2))
        else:
            mins = [v[0] for v in val]
            maxs = [v[1] for v in val]
            # Round activation ranges
            final_ranges[key] = (round(min(mins), 2), round(max(maxs), 2))
            
    return final_ranges

# 7. Main Execution
def main():
    # 1. Use a fixed directory for models to allow reuse
    exp_dir = "mbe_models/toy_mlp_calibrated"
    plot_dir = "plots/toy_mlp_calibrated"
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"=== Starting Experiment ===")
    print(f"Models directory: {exp_dir}")
    print(f"Plots directory: {plot_dir}")
    
    # Data
    train_loader, test_loader = generate_data()
    
    # 1. Train ANN
    ann_model = ToyTransformerMLP().to(DEVICE)
    train_ann(ann_model, train_loader, epochs=10)
    ann_acc, _ = evaluate_with_metrics(ann_model, test_loader)
    print(f"\nFinal ANN Test Accuracy: {ann_acc:.2f}%")
    
    # 2. Calibration
    print("\nStarting Calibration...")
    final_ranges = calibrate_ranges(ann_model, train_loader)
    for k, v in final_ranges.items():
        print(f"  {k:10}: [{v[0]:.4f}, {v[1]:.4f}]")
    
    # 3. SNN Conversion & Fitting
    print("\nStarting SNN Conversion and MBE Fitting...")
    snn_model = ToyTransformerSNN(ann_model, timesteps=16, num_basis=8).to(DEVICE)
    
    # Fit MBE Activation (GELU) with specific paths
    snn_model.gelu1.model_path = os.path.join(exp_dir, "gelu1.pth")
    snn_model.gelu1.fit(x_range=final_ranges['gelu1_in'], epochs=1000)
    
    snn_model.gelu2.model_path = os.path.join(exp_dir, "gelu2.pth")
    snn_model.gelu2.fit(x_range=final_ranges['gelu2_in'], epochs=1000)
    
    # For MBEMultiplier Identity model, use fixed range [-1, 1] 
    # since we use per-layer normalization in MBELinear.
    id_range = (-1.0, 1.0)
    mbe_id_path = os.path.join(exp_dir, "id_mbe.pth")
    
    if os.path.exists(mbe_id_path):
        print(f"Loading pre-trained Identity MBE from {mbe_id_path}")
        mbe_id = MBENeuron.load(mbe_id_path)
    else:
        print(f"\nTraining Identity MBE for Multipliers with fixed range: {id_range}")
        mbe_id = train_mbe_neuron(
            target_func=lambda x: x,
            x_range=id_range,
            num_samples=10000,
            num_epochs=1000,
            lr=0.01,
            num_basis=8,
            timesteps=16,
            alpha=1.0
        )
        mbe_id.save(mbe_id_path)
    
    snn_model.fc1.initialize_multiplier(mbe_id)
    snn_model.fc2.initialize_multiplier(mbe_id)
    snn_model.fc3.initialize_multiplier(mbe_id)
    
    # 4. SNN Evaluation with Metrics
    print("\nEvaluating SNN Metrics...")
    snn_acc, avg_spikes = evaluate_with_metrics(snn_model, test_loader)
    
    # Calculate MACs for ANN
    # fc1: 20*64, fc2: 64*64, fc3: 64*3
    ann_macs = (20*64) + (64*64) + (64*3)
    # LayerNorm and GELU also have MACs, but Linear is dominant.
    
    # SOPs (Synaptic Operations)
    # In MBE, each spike in MBEMultiplier or MBENeuron leads to an addition.
    snn_sops = avg_spikes # Simplified for this toy model
    
    # Energy Calculation (45nm CMOS)
    e_mac = 3.1 # pJ
    e_sop = 0.1 # pJ
    ann_energy = ann_macs * e_mac
    snn_energy = snn_sops * e_sop
    energy_saving = (1 - snn_energy / ann_energy) * 100

    print(f"\n--- Performance Summary ---")
    print(f"Final ANN Test Accuracy: {ann_acc:.2f}%")
    print(f"Final SNN Test Accuracy: {snn_acc:.2f}%")
    print(f"Accuracy Drop: {ann_acc - snn_acc:.2f}%")
    print(f"Avg Spikes per Sample: {avg_spikes:.2f}")
    print(f"Energy Saving (Estimated): {energy_saving:.2f}%")
    
    # 5. Visualizations
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Accuracy Comparison
    plt.subplot(1, 2, 1)
    models = ['ANN', 'SNN']
    accs = [ann_acc, snn_acc]
    plt.bar(models, accs, color=['blue', 'red'])
    plt.ylim(0, 110)
    plt.title('Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    
    # Plot 2: Energy Comparison
    plt.subplot(1, 2, 2)
    energies = [ann_energy, snn_energy]
    plt.bar(['ANN (MAC)', 'SNN (SOP)'], energies, color=['blue', 'green'])
    plt.title('Estimated Energy Consumption (pJ)')
    plt.ylabel('Energy (pJ)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "efficiency_report.png"))
    print(f"\nVerification Complete. Plots saved to {plot_dir}")

if __name__ == "__main__":
    main()
