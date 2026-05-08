import torch
import torch.nn as nn
import torch.optim as optim
from MBE_neurons import MBENeuron

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def evaluate_with_metrics(model, loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_sops = 0
    
    # We will use a hook to count SOPs correctly
    def get_sop_hook(fan_out=1):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                # s_seq: (T, N, Batch, Dim)
                s_seq = output[1]
                spikes = s_seq.sum().item()
                # Synaptic Operations = Spikes * Fan-out
                # For activations, fan_out is 1. For Linear, it's out_features.
                nonlocal total_sops
                total_sops += spikes * fan_out
        return hook

    hooks = []
    # Identify modules and attach appropriate hooks
    from mbe_modules import MBELinear, MBEActivation, MBELayerNorm, MBESoftmax
    
    for name, m in model.named_modules():
        if isinstance(m, MBELinear):
            # For Linear, SOPs = Input Spikes * Out Features
            # We hook the internal mbe_id neuron
            if m.mbe_id is not None:
                hooks.append(m.mbe_id.register_forward_hook(get_sop_hook(fan_out=m.out_features)))
        elif isinstance(m, (MBEActivation, MBELayerNorm, MBESoftmax)):
            # For these, we search for internal MBENeurons and hook them with fan_out=1
            for sub_m in m.modules():
                if isinstance(sub_m, MBENeuron):
                    # Avoid double hooking if it's already handled
                    hooks.append(sub_m.register_forward_hook(get_sop_hook(fan_out=1)))

    print(f"Starting SNN Evaluation on {len(loader.dataset)} samples...")
    batch_count = 0
    total_batches = len(loader)
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Only count SOPs for a subset of batches to save time (it's representative)
            if batch_count < 20:
                outputs = model(batch_x)
            else:
                # Remove hooks after enough samples
                if batch_count == 20:
                    for h in hooks: h.remove()
                    hooks = []
                    print(" (SOP sampling complete, continuing with inference only...)")
                outputs = model(batch_x)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            batch_count += 1
            if batch_count % 10 == 0 or batch_count == total_batches:
                print(f" Batch [{batch_count}/{total_batches}] complete...")

    for h in hooks: h.remove()
    acc = 100 * correct / total
    avg_sops_per_sample = total_sops / total
    
    return acc, avg_sops_per_sample

def train_ann(model, train_loader, device, epochs=10, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        acc, _ = evaluate_with_metrics(model, train_loader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Train Acc: {acc:.2f}%")
