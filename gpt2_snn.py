import os
import torch
import math
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers.pytorch_utils import Conv1D
from datasets import load_dataset
import torch.nn as nn

from MBE_neurons import train_mbe_neuron, MBENeuron
from mbe_modules import MBEConv1D, MBEGELU
from approximate_layernorm import MBELayerNorm
from approximate_softmax import MBESoftmax

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_gpt2_ann_energy(seq_len=512):
    """Calculate theoretical ANN energy for GPT-2 Small per sequence."""
    D = 768
    V = 50257
    num_layers = 12
    
    # MACs per token per layer
    macs_attn_qkv = 3 * D * D
    macs_attn_proj = D * D
    macs_mlp_fc = 4 * D * D
    macs_mlp_proj = 4 * D * D
    macs_attn_scores = 2 * seq_len * D # QK^T and AV
    macs_layer = macs_attn_qkv + macs_attn_proj + macs_mlp_fc + macs_mlp_proj + macs_attn_scores
    
    macs_total_linear = num_layers * macs_layer * seq_len
    macs_lm_head = D * V * seq_len
    
    total_macs = macs_total_linear + macs_lm_head
    e_linear = total_macs * 4.6 # pJ
    
    # LayerNorm
    e_ln = seq_len * (2 * num_layers + 1) * (14.7 * D + 41.8)
    
    # GELU
    e_gelu = seq_len * num_layers * (4 * D * 65.4)
    
    # Softmax
    e_softmax_attn = num_layers * 12 * seq_len * (58.0 * seq_len - 0.9)
    e_softmax_head = seq_len * (58.0 * V - 0.9)
    e_softmax = e_softmax_attn + e_softmax_head
    
    total_energy_pj = e_linear + e_ln + e_gelu + e_softmax
    
    return {
        'total_pj': total_energy_pj,
        'e_linear': e_linear,
        'e_nonlinear': e_ln + e_gelu + e_softmax
    }

def calculate_perplexity_and_metrics(model, tokenizer, dataset, device, stride=512, max_length=1024, num_samples=None, count_sops=False):
    model.eval()
    encodings = tokenizer("\n\n".join(dataset['text']), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    if num_samples is not None:
        seq_len = min(seq_len, num_samples * max_length)

    total_sops = 0
    total_spikes = 0
    total_elements = 0
    hooks = []

    if count_sops:
        def get_sop_hook(fan_out=1):
            def hook(module, input, output):
                nonlocal total_sops, total_spikes, total_elements
                if isinstance(output, tuple):
                    s_seq = output[1]
                    num_spikes = s_seq.sum().item()
                    total_spikes += num_spikes
                    total_elements += s_seq.numel()
                    total_sops += num_spikes * fan_out
            return hook

        for name, m in model.named_modules():
            if isinstance(m, MBEConv1D):
                hooks.append(m.mbe.register_forward_hook(get_sop_hook(fan_out=m.nf)))

    nlls = []
    prev_end_loc = 0
    seq_count = 0
    
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating PPL & Metrics"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        seq_count += 1
        if end_loc == seq_len:
            break

    for h in hooks:
        h.remove()

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    avg_sops_per_seq = total_sops / seq_count if seq_count > 0 else 0
    avg_eta = (total_spikes / total_elements) if total_elements > 0 else 0
    
    return ppl, avg_sops_per_seq, avg_eta

def calibrate_model(model, tokenizer, dataset, device, num_samples=10, seq_length=256):
    model.eval()
    ranges = {}

    def get_hook(name):
        def hook(module, input, output):
            val = input[0].detach()
            v_min, v_max = val.min().item(), val.max().item()
            if name not in ranges:
                ranges[name] = [v_min, v_max]
            else:
                ranges[name][0] = min(ranges[name][0], v_min)
                ranges[name][1] = max(ranges[name][1], v_max)
        return hook

    hooks = []
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.ln_1.register_forward_hook(get_hook(f"layer_{i}_ln_1")))
        hooks.append(block.attn.c_attn.register_forward_hook(get_hook(f"layer_{i}_c_attn")))
        hooks.append(block.attn.c_proj.register_forward_hook(get_hook(f"layer_{i}_attn_c_proj")))
        hooks.append(block.mlp.act.register_forward_hook(get_hook(f"layer_{i}_gelu")))
        hooks.append(block.mlp.c_fc.register_forward_hook(get_hook(f"layer_{i}_c_fc")))
        hooks.append(block.mlp.c_proj.register_forward_hook(get_hook(f"layer_{i}_mlp_c_proj")))
    hooks.append(model.transformer.ln_f.register_forward_hook(get_hook("ln_f")))

    print(f"Calibrating over {num_samples} sequences of length {seq_length}...")
    encodings = tokenizer("\n\n".join(dataset['text']), return_tensors="pt").input_ids
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            start = i * seq_length
            end = start + seq_length
            if end > encodings.size(1): break
            input_ids = encodings[:, start:end].to(device)
            model(input_ids)

    for h in hooks:
        h.remove()
    return ranges

def replace_modules_with_mbe(model, ranges, args, device):
    print(f"Replacing modules with MBE variants (T={args.timesteps}, N={args.num_basis})...")
    model_dir = "mbe_models/gpt2"
    os.makedirs(model_dir, exist_ok=True)
    
    id_range = (-1.0, 1.0)
    id_path = os.path.join(model_dir, f"mbe_id_T{args.timesteps}_N{args.num_basis}.pth")
    if os.path.exists(id_path):
        mbe_id = MBENeuron.load(id_path)
    else:
        print("Training Global Identity MBE for Conv1D...")
        mbe_id = train_mbe_neuron(lambda x: x, x_range=id_range, num_basis=args.num_basis, timesteps=args.timesteps, alpha=1.0, num_epochs=args.epochs, l1_spike_weight=args.l1_spike_weight, target_loss=args.target_loss, patience=args.patience)
        mbe_id.save(id_path)
    
    mbe_id.to(device)

    for i, block in enumerate(model.transformer.h):
        print(f" Converting Layer {i}...")
        
        # LayerNorm
        if args.replace_ln:
            ln1_r = ranges[f"layer_{i}_ln_1"]
            mbe_ln1 = MBELayerNorm(normalized_shape=block.ln_1.normalized_shape[0], timesteps=args.timesteps, num_basis=args.num_basis)
            mbe_ln1.load_from_standard_layernorm(block.ln_1)
            block.ln_1 = mbe_ln1.to(device)

        # GELU
        if args.replace_gelu:
            gelu_r = ranges[f"layer_{i}_gelu"]
            gelu_path = os.path.join(model_dir, f"mbe_gelu_L{i}_T{args.timesteps}_N{args.num_basis}.pth")
            mbe_gelu = MBEGELU(timesteps=args.timesteps, num_basis=args.num_basis, model_path=gelu_path)
            mbe_gelu.fit(x_range=(gelu_r[0], gelu_r[1]), epochs=args.epochs, l1_spike_weight=args.l1_spike_weight, target_loss=args.target_loss, patience=args.patience)
            block.mlp.act = mbe_gelu.to(device)

        # Conv1D
        if args.replace_conv1d:
            import copy
            def replace_conv1d(conv1d_module, name):
                mbe_c = MBEConv1D(nf=conv1d_module.nf, nx=conv1d_module.weight.shape[0], 
                                  num_basis=args.num_basis, timesteps=args.timesteps)
                mbe_c.load_from_standard_conv1d(conv1d_module)
                mbe_c.initialize_multiplier(copy.deepcopy(mbe_id))
                return mbe_c.to(device)

            block.attn.c_attn = replace_conv1d(block.attn.c_attn, f"layer_{i}_c_attn")
            block.attn.c_proj = replace_conv1d(block.attn.c_proj, f"layer_{i}_attn_c_proj")
            block.mlp.c_fc = replace_conv1d(block.mlp.c_fc, f"layer_{i}_c_fc")
            block.mlp.c_proj = replace_conv1d(block.mlp.c_proj, f"layer_{i}_mlp_c_proj")
        
    print("SNN Conversion Complete.")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gpt2')
    parser.add_argument('--evaluate_ann_only', action='store_true')
    parser.add_argument('--calibrate_only', action='store_true')
    parser.add_argument('--evaluate_snn', action='store_true')
    parser.add_argument('--timesteps', type=int, default=16)
    parser.add_argument('--num_basis', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2000, help="Training epochs for MBE on-the-fly")
    parser.add_argument('--l1_spike_weight', type=float, default=0.0, help="L1 regularization for spike suppression")
    parser.add_argument('--target_loss', type=float, default=1e-4, help="Target MSE for early stopping")
    parser.add_argument('--patience', type=int, default=1000, help="Patience for early stopping")
    parser.add_argument('--replace_conv1d', action='store_true', default=True)
    parser.add_argument('--replace_gelu', action='store_true', default=True)
    parser.add_argument('--replace_ln', action='store_true', default=True)
    parser.add_argument('--no_replace_conv1d', action='store_false', dest='replace_conv1d')
    parser.add_argument('--no_replace_gelu', action='store_false', dest='replace_gelu')
    parser.add_argument('--no_replace_ln', action='store_false', dest='replace_ln')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_id)
    model = GPT2LMHeadModel.from_pretrained(args.model_id).to(device)

    print("Loading WikiText-103 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')

    if args.evaluate_ann_only:
        print("Evaluating baseline ANN Perplexity...")
        ppl, _, _ = calculate_perplexity_and_metrics(model, tokenizer, dataset, device, num_samples=10, count_sops=False)
        print(f"Baseline ANN PPL: {ppl:.2f}")
        return

    ranges = calibrate_model(model, tokenizer, dataset, device, num_samples=5)
    
    if args.calibrate_only:
        print("Calibration completed.")
        return

    if args.evaluate_snn:
        snn_model = replace_modules_with_mbe(model, ranges, args, device)
        
        print("Evaluating SNN Perplexity & Metrics...")
        snn_ppl, avg_sops, avg_eta = calculate_perplexity_and_metrics(snn_model, tokenizer, dataset, device, num_samples=10, count_sops=True)
        
        # --- Efficiency Calculation ---
        T = args.timesteps
        N = args.num_basis
        stride = 512
        
        # 1. ANN Energy Baseline
        energy_ann = calculate_gpt2_ann_energy(seq_len=stride)
        total_ann_energy = energy_ann['total_pj']
        
        # 2. SNN Energy
        # Inference Energy (Pure SOPs at 0.9 pJ)
        e_snn_inference = avg_sops * 0.9
        
        # Pre-computation Energy (One-time cost to cache Weff = W * dn)
        # 12 layers * (3*D^2 + D^2 + 4*D^2 + 4*D^2) + D*V
        D = 768
        V = 50257
        num_weights = 12 * (12 * D * D) + (D * V)
        e_snn_precompute = (num_weights * N * T) * 3.7
        
        # Total SNN Energy (Amortized over K tokens, say K=1,000,000)
        K = 1000000 
        total_snn_energy = (e_snn_precompute / K) + e_snn_inference
        
        cdcer = (1 - total_snn_energy / total_ann_energy) * 100

        print(f"\n--- GPT-2 Small SNN Evaluation Report ---")
        print(f"Target SNN PPL (from paper): 23.41")
        print(f"Achieved SNN PPL: {snn_ppl:.2f}")
        print(f"\n--- Efficiency Analysis (Per Sequence of {stride} tokens) ---")
        print(f"Avg Firing Rate (eta): {avg_eta*100:.2f}%")
        print(f"ANN Total Energy: {total_ann_energy/1e6:.2f} uJ")
        print(f"SNN Total Energy: {total_snn_energy/1e6:.2f} uJ (Pre-compute amortized)")
        print(f"CDCER Score: {cdcer:.2f}%")

if __name__ == "__main__":
    main()
