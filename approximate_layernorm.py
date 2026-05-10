"""
MBE Neuron-based Layer Normalization Approximation  (v4)
=========================================================
LN(x_i) = gamma * (x_i - mu) / sqrt(var + eps) + beta

핵심 수정 사항 (v4):
  - Step 1: MBEMultiplier(y,y) → MBE가 y² 함수 직접 근사
            (identity MBE의 대범위 수렴 실패 문제 해소)
  - Step 3: MBEMultiplier 입력을 [-1,1]로 정규화 후 곱셈
            (softmax에서 [0,1]로 성공한 전략 적용)

1/n 흡수 수식:
  1/sqrt(sum_sq/n + eps) = sqrt(n) * 1/sqrt(sum_sq + n*eps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from MBE_neurons import MBENeuron, train_mbe_neuron
from approximate_fp_mult import MBEMultiplier

def set_seed(seed):
    """Fix seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'mbe_models', 'layernorm')


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
def _load_or_train(model_path, target_func, x_range, tag,
                   num_basis=8, timesteps=16, num_epochs=3000, lr=0.005, tv_weight=0.0,
                   l1_spike_weight=0.0, target_loss=1e-4, patience=1000):
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(model_path):
        print(f"[{tag}] Loading from {model_path}")
        return MBENeuron.load(model_path)
    print(f"[{tag}] Training on range {x_range} ...")
    x_probe = torch.linspace(x_range[0], x_range[1], 4000)
    alpha = float(target_func(x_probe).abs().max().item())
    alpha = max(alpha, 1e-3)
    model = train_mbe_neuron(
        target_func=target_func, x_range=x_range, num_samples=10000,
        num_epochs=num_epochs, lr=lr, tv_weight=tv_weight,
        num_basis=num_basis, timesteps=timesteps, alpha=alpha,
        l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
    )
    model.save(model_path)
    print(f"[{tag}] Saved to {model_path}")
    return model


def _make_id_multiplier(model_path, x_range, tag,
                        num_basis=8, timesteps=16,
                        num_epochs=5000, lr=0.001, tv_weight=0.05,
                        l1_spike_weight=0.0, target_loss=1e-4, patience=1000):
    """Identity MBE 학습 후 MBEMultiplier 반환."""
    mbe_id = _load_or_train(
        model_path, lambda x: x, x_range=x_range, tag=tag,
        num_basis=num_basis, timesteps=timesteps,
        num_epochs=num_epochs, lr=lr, tv_weight=tv_weight,
        l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
    )
    return MBEMultiplier(mbe_id_model=mbe_id)


def _fp_decompose_even_exp(v: torch.Tensor):
    """v > 0 를 V = M_adj * 2^{E_adj} 로 분해. E_adj 는 항상 짝수."""
    v_safe = v.abs().clamp(min=1e-38)
    E_raw  = torch.floor(torch.log2(v_safe))
    M_raw  = v_safe / torch.pow(2.0, E_raw)
    is_odd = (E_raw % 2).abs() > 0.5
    E_adj  = torch.where(is_odd, E_raw - 1, E_raw)
    M_adj  = torch.where(is_odd, M_raw * 2.0, M_raw)
    return M_adj, E_adj


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 - x^2 직접 근사: MBENeuron(a) ≈ a^2
# ─────────────────────────────────────────────────────────────────────────────
class MBESquare(nn.Module):
    """
    centered = x - mu  (선형)
    centered^2 ≈ MBE_Direct_Sq(centered/s) * s^2

    [Per-sample 정규화 전략]
    y -> a = y / max(|y|) in [-1, 1]
    MBE 뉴런이 [-1, 1] 범위에서 a^2 함수를 직접 근사함.
    """

    def __init__(self, num_basis=8, timesteps=16, epochs=5000, lr=0.001, tv_weight=0.05, l1_spike_weight=0.0, target_loss=1e-4, patience=1000):
        super().__init__()
        # x^2 직접 근사 (정규화된 범위 [-1, 1]에서 학습)
        model_name = f"mbe_ln_direct_sq_T{timesteps}_N{num_basis}.pth"
        self.mbe_sq = _load_or_train(
            model_path=os.path.join(MODEL_DIR, model_name),
            target_func=lambda x: x**2,
            x_range=(-1.0, 1.0),
            tag='MBESquare',
            num_basis=num_basis, timesteps=timesteps,
            num_epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )

    def forward(self, x: torch.Tensor, dim: int = -1):
        mu = x.mean(dim=dim, keepdim=True)
        centered = x - mu

        # per-sample 정규화 → [-1, 1]
        s = centered.abs().amax(dim=dim, keepdim=True).detach().clamp(min=1e-6)
        a = (centered / s).clamp(-1, 1)

        # 직접 근사 뉴런 통과
        orig_shape = a.shape
        a_flat = a.reshape(-1, 1)
        
        self.mbe_sq.eval()
        with torch.no_grad():
            sq_a_flat = self.mbe_sq(a_flat)
        
        centered_sq = sq_a_flat.reshape(orig_shape).clamp(min=0.0)

        # 스케일 복원: a^2 * s^2 = centered^2
        centered_sq = centered_sq * (s ** 2)

        sum_sq = centered_sq.sum(dim=dim, keepdim=True)
        return centered, sum_sq


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – 역제곱근 근사
# ─────────────────────────────────────────────────────────────────────────────
class MBEInvSqrt(nn.Module):
    """
    V = M_adj * 2^{E_adj} (E_adj 짝수)
    1/sqrt(V) = 1/sqrt(M_adj) * 2^{-E_adj/2}
    MBE: M_adj ∈ [1, 4) → 1/sqrt(M_adj)
    """

    def __init__(self, num_basis=8, timesteps=16, model_path=None, epochs=3000, lr=0.005, tv_weight=0.05, l1_spike_weight=0.0, target_loss=1e-4, patience=1000):
        super().__init__()
        if model_path is None:
            model_name = f"mbe_ln_inv_sqrt_mantissa_T{timesteps}_N{num_basis}.pth"
            model_path = os.path.join(MODEL_DIR, model_name)
        self.mbe_invsqrt = _load_or_train(
            model_path,
            target_func=lambda x: 1.0 / torch.sqrt(x.clamp(min=1e-6)),
            x_range=(1.0, 4.0 - 1e-3),
            tag='MBEInvSqrt-mantissa',
            num_basis=num_basis, timesteps=timesteps,
            num_epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        M_adj, E_adj = _fp_decompose_even_exp(v)
        orig_shape = M_adj.shape
        self.mbe_invsqrt.eval()
        with torch.no_grad():
            inv_sqrt_M = self.mbe_invsqrt(
                M_adj.reshape(-1, 1)
            ).reshape(orig_shape)
        shift = torch.pow(2.0, -E_adj / 2.0)
        return inv_sqrt_M * shift


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – FP 곱셈 스케일링 (정규화 전략)
# ─────────────────────────────────────────────────────────────────────────────
class MBEFPScaling(nn.Module):
    """
    ① x_norm = MBEMult(centered, inv_sqrt_var)
    ② out    = MBEMult(gamma, x_norm) + beta

    [입력 정규화 전략]
    centered, inv_sqrt_var 를 각각 최대절대값으로 나눠 [-1,1]로 스케일링.
    Identity MBE 가 [-1,1] 에서 수렴 (softmax [0,1] 성공 사례 참조).
    곱셈 후 스케일 복원: result * s_c * s_i
    """

    def __init__(self, normalized_dim: int, num_basis=8, timesteps=16, epochs=5000, lr=0.001, tv_weight=0.05, l1_spike_weight=0.0, target_loss=1e-4, patience=1000):
        super().__init__()
        # 정규화 후 [-1, 1] 범위의 identity MBE
        norm_name = f"mbe_ln_id_norm_T{timesteps}_N{num_basis}.pth"
        gamma_name = f"mbe_ln_id_gamma_T{timesteps}_N{num_basis}.pth"
        
        self.mult_norm = _make_id_multiplier(
            model_path=os.path.join(MODEL_DIR, norm_name),
            x_range=(-1.0, 1.0),
            tag='MBEId-norm',
            num_basis=num_basis, timesteps=timesteps,
            num_epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )
        self.mult_gamma = _make_id_multiplier(
            model_path=os.path.join(MODEL_DIR, gamma_name),
            x_range=(-1.0, 1.0),
            tag='MBEId-gamma',
            num_basis=num_basis, timesteps=timesteps,
            num_epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )
        self.gamma = nn.Parameter(torch.ones(normalized_dim))
        self.beta  = nn.Parameter(torch.zeros(normalized_dim))

    def forward(self, centered: torch.Tensor,
                inv_sqrt_var: torch.Tensor):
        """
        [Per-sample 정규화 전략]
        s_c, s_n 을 keepdim=True 로 각 샘플별로 독립 계산.
        전역 max() 사용 시 한 샘플의 outlier가 다른 샘플 정규화를 왜곡
        → 모든 값이 0으로 눌려 MBE가 같은 출력(-0.8449 반복) 생성하는 문제 해소.
        """
        # ── per-sample 스케일 참조값 (dim=-1 keepdim) ──
        # centered: (..., n) → s_c: (..., 1), 각 샘플 독립
        s_c = centered.abs().amax(dim=-1, keepdim=True).detach().clamp(min=1e-6)
        # inv_sqrt_var: (..., 1) → 이미 per-sample scalar이므로 그대로 사용
        s_i = inv_sqrt_var.abs().detach().clamp(min=1e-6)

        # ① 정규화 → [-1,1]  (각 샘플 독립)
        a = (centered / s_c).clamp(-1, 1)           # (..., n)
        b = (inv_sqrt_var / s_i).clamp(-1, 1).expand_as(a)  # (..., n)

        # ② MBEMult(a, b) ≈ a * b
        x_norm_scaled = self.mult_norm(a, b)

        # ③ 스케일 복원: (a*b) * s_c * s_i = centered * inv_sqrt_var = x_norm
        x_norm = x_norm_scaled * s_c * s_i

        # ④ gamma * x_norm (per-sample 정규화)
        s_n = x_norm.abs().amax(dim=-1, keepdim=True).detach().clamp(min=1e-6)
        s_g = self.gamma.abs().max().detach().clamp(min=1e-6)
        n_scaled = (x_norm / s_n).clamp(-1, 1)
        g_scaled = (self.gamma / s_g).clamp(-1, 1).expand_as(n_scaled)
        out_scaled = self.mult_gamma(n_scaled, g_scaled)
        out = out_scaled * s_n * s_g

        # ⑤ + beta (선형)
        out = out + self.beta
        return out, x_norm


# ─────────────────────────────────────────────────────────────────────────────
# 통합 MBELayerNorm
# ─────────────────────────────────────────────────────────────────────────────
class MBELayerNorm(nn.Module):
    """
    1/sqrt(sum_sq/n + eps) = sqrt(n) * 1/sqrt(sum_sq + n*eps)
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5,
                 num_basis: int = 8, timesteps: int = 16,
                 epochs: int = 3000, lr: float = 0.005, tv_weight: float = 0.05,
                 l1_spike_weight: float = 0.0, target_loss: float = 1e-4, patience: int = 1000):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.step1 = MBESquare(
            num_basis=num_basis, timesteps=timesteps, 
            epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )
        self.step2 = MBEInvSqrt(
            num_basis=num_basis, timesteps=timesteps,
            epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )
        self.step3 = MBEFPScaling(
            normalized_dim=normalized_shape,
            num_basis=num_basis, timesteps=timesteps,
            epochs=epochs, lr=lr, tv_weight=tv_weight,
            l1_spike_weight=l1_spike_weight, target_loss=target_loss, patience=patience
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        centered, sum_sq = self.step1(x, dim=-1)

        v = sum_sq + sum_sq.new_full(sum_sq.shape, n * self.eps)
        inv_sqrt  = self.step2(v)
        inv_sqrt_var = inv_sqrt * math.sqrt(n)

        out, _ = self.step3(centered, inv_sqrt_var)
        return out

    def load_from_standard_layernorm(self, ln: nn.LayerNorm):
        with torch.no_grad():
            if ln.weight is not None:
                self.step3.gamma.copy_(ln.weight)
            if ln.bias is not None:
                self.step3.beta.copy_(ln.bias)
        print("[MBELayerNorm] gamma/beta loaded from nn.LayerNorm.")


# ─────────────────────────────────────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────────────────────────────────────
def test_invsqrt(args=None):
    print("\n" + "=" * 60)
    print("  [TEST] MBEInvSqrt")
    print("=" * 60)
    num_basis = args.num_basis if args else 8
    timesteps = args.timesteps if args else 16
    m = MBEInvSqrt(num_basis=num_basis, timesteps=timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)
    
    v_test = torch.linspace(0.01, 32.0, 200)
    true_v = 1.0 / torch.sqrt(v_test)
    approx = m(v_test)
    
    print(f"  {'V':>8} | {'true':>12} | {'approx':>12}")
    print("  " + "-" * 40)
    for i in [0, 50, 100, 150, 199]:
        print(f"  {v_test[i].item():8.3f} | {true_v[i].item():12.6f} | {approx[i].item():12.6f}")

    if args and args.plot:
        plot_dir = os.path.join('plots', 'layernorm')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(v_test.numpy(), true_v.numpy(), 'b-', label='True $1/\sqrt{V}$')
        plt.plot(v_test.numpy(), approx.numpy(), 'r--', label='MBE Approx')
        plt.title(f"MBE Inverse Square Root (T={timesteps})")
        plt.xlabel("V")
        plt.ylabel("$1/\sqrt{V}$")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(plot_dir, 'mbe_ln_step2_invsqrt.png')
        plt.savefig(save_path)
        print(f"  Plot saved to {save_path}")


def test_step_by_step(args=None):
    print("\n" + "=" * 60)
    print("  [TEST] Step-by-Step Data Flow")
    print("=" * 60)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    n = x.shape[-1]
    eps = 1e-5
    
    num_basis = args.num_basis if args else 8
    timesteps = args.timesteps if args else 16

    sq  = MBESquare(num_basis=num_basis, timesteps=timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)
    isq = MBEInvSqrt(num_basis=num_basis, timesteps=timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)

    centered, sum_sq = sq(x, dim=-1)
    true_sum_sq = ((x - x.mean()) ** 2).sum().item()
    true_var = x.var(dim=-1, unbiased=False).item() + eps

    print(f"\n  x        = {x.tolist()}")
    print(f"  mu       = {x.mean().item():.4f}")
    print(f"  centered = {centered.tolist()}")
    print(f"  sum_sq   MBE = {sum_sq.item():.6f}  (true: {true_sum_sq:.6f})")

    v = sum_sq + n * eps
    inv_sqrt = isq(v) * math.sqrt(n)
    true_inv = 1.0 / math.sqrt(true_var)
    print(f"\n  1/sqrt(var) MBE  = {inv_sqrt.item():.6f}")
    print(f"  1/sqrt(var) true = {true_inv:.6f}")
    print(f"  Relative Error   = {abs(inv_sqrt.item() - true_inv) / true_inv:.4f}")


def test_layernorm(args):
    print("\n" + "=" * 60)
    print("  [TEST] MBELayerNorm vs nn.LayerNorm")
    print("=" * 60)
    B, D = 100, 8
    x = torch.randn(B, D) * 2.0

    std_ln   = nn.LayerNorm(D)
    true_out = std_ln(x)

    mbe_ln = MBELayerNorm(
        normalized_shape=D, eps=1e-5, 
        num_basis=args.num_basis, timesteps=args.timesteps,
        epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight
    )
    mbe_ln.load_from_standard_layernorm(std_ln)

    mbe_ln.eval()
    with torch.no_grad():
        approx_out = mbe_ln(x)

    mse = F.mse_loss(approx_out, true_out).item()
    print("\n  True LayerNorm (First 2 samples):")
    for row in true_out[:2]:
        print("   ", [f"{v:7.4f}" for v in row.tolist()])
    print("  MBE LayerNorm (First 2 samples):")
    for row in approx_out[:2]:
        print("   ", [f"{v:7.4f}" for v in row.tolist()])
    print(f"\n  MSE           : {mse:.8f}")

    if args.plot:
        plot_dir = os.path.join('plots', 'layernorm')
        os.makedirs(plot_dir, exist_ok=True)
        
        t_flat = true_out.detach().numpy()
        a_flat = approx_out.detach().numpy()
        
        # 1. Line Plot: 특정 샘플(Batch 0)의 요소별 비교 (별도 저장)
        plt.figure(figsize=(10, 6))
        indices = np.arange(D)
        plt.plot(indices, t_flat[0], 'b-o', label='True LayerNorm', alpha=0.7, markersize=8)
        plt.plot(indices, a_flat[0], 'r--X', label='MBE LayerNorm', alpha=0.8, markersize=8)
        for idx in indices:
            plt.text(idx, t_flat[0][idx] + 0.05, f'{t_flat[0][idx]:.2f}', color='blue', ha='center', fontsize=8)
            plt.text(idx, a_flat[0][idx] - 0.15, f'{a_flat[0][idx]:.2f}', color='red', ha='center', fontsize=8)
        plt.title(f"LayerNorm Element-wise Comparison (Batch 0)\nMSE: {mse:.6f}", fontsize=12)
        plt.xlabel("Feature Index")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        save_path_val = os.path.join(plot_dir, 'mbe_layernorm_val_comparison.png')
        plt.savefig(save_path_val)
        plt.close()

        # 2. Histogram: 전체적인 분포 일치도 확인 (별도 저장)
        plt.figure(figsize=(10, 6))
        
        # 오버랩(Histogram Intersection) 계산
        bins = 30
        hist_range = (min(t_flat.min(), a_flat.min()), max(t_flat.max(), a_flat.max()))
        h_true, _ = np.histogram(t_flat, bins=bins, range=hist_range, density=True)
        h_approx, _ = np.histogram(a_flat, bins=bins, range=hist_range, density=True)
        
        # 교차 면적 계산 (두 히스토그램 중 작은 값들의 합 * bin_width)
        bin_width = (hist_range[1] - hist_range[0]) / bins
        overlap = np.sum(np.minimum(h_true, h_approx)) * bin_width
        overlap_pct = overlap * 100

        plt.hist(t_flat.flatten(), bins=bins, range=hist_range, alpha=0.5, label='True LN Distribution', color='#3498db', density=True)
        plt.hist(a_flat.flatten(), bins=bins, range=hist_range, alpha=0.4, label='MBE LN Distribution', color='#e74c3c', density=True)
        
        plt.title(f"Overall LayerNorm Output Distribution\nDistribution Overlap: {overlap_pct:.2f}%", fontsize=12)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        save_path_dist = os.path.join(plot_dir, 'mbe_layernorm_dist_comparison.png')
        plt.savefig(save_path_dist)
        plt.close()

        print(f"  Plots saved to {plot_dir}:")
        print(f"    - {os.path.basename(save_path_val)}")
        print(f"    - {os.path.basename(save_path_dist)} (Overlap: {overlap_pct:.2f}%)")


def test_components(args):
    """LayerNorm에 필요한 각 MBE 구성요소의 정확도를 독립적으로 평가 및 시각화."""
    print("\n" + "=" * 70)
    print("  [COMPONENT TEST] LayerNorm MBE 구성요소 정확도 평가")
    print("=" * 70)
    plot_dir = os.path.join('plots', 'layernorm')
    os.makedirs(plot_dir, exist_ok=True)

    # ── 1. MBESquare: y² 근사 ─────────────────────────────────────
    print("\n  [1] MBESquare: y^2 근사 (직접 근사 방식)")
    sq = MBESquare(num_basis=args.num_basis, timesteps=args.timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)
    y_test = torch.linspace(-3.0, 3.0, 200)
    true_sq = y_test ** 2
    
    sq.eval()
    with torch.no_grad():
        # MBESquare 내부의 정규화 로직을 타기 위해 (1, N) 형태로 구성
        y_input = y_test.unsqueeze(0)
        _, sum_sq_val = sq(y_input, dim=-1) # 이건 sum_sq를 반환하므로 테스트용으로는 부적합
        
        # 개별 값 테스트를 위해 직접 뉴런 호출
        s_test = 3.0 # 정규화 스케일 가정
        a_test = (y_test / s_test).unsqueeze(1)
        approx_sq = sq.mbe_sq(a_test).squeeze() * (s_test ** 2)

    print(f"  MSE: {F.mse_loss(approx_sq, true_sq).item():.6f}")

    if args.plot:
        plt.figure(figsize=(8, 6))
        plt.plot(y_test.numpy(), true_sq.numpy(), 'b-', label='True $y^2$')
        plt.plot(y_test.numpy(), approx_sq.numpy(), 'r--', label='MBE Approx')
        plt.title(f"MBE Square Approximation (T={args.timesteps})")
        plt.xlabel("y")
        plt.ylabel("$y^2$")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'mbe_ln_step1_sq.png'))

    # ── 2. Identity MBE: f(x)=x 근사 ─────────────────────────────────
    print("\n  [2] Identity MBE on [-1, 1] (MBEMultiplier 내부 핵심)")
    scaling = MBEFPScaling(normalized_dim=1, num_basis=args.num_basis, timesteps=args.timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)
    x_id = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)
    scaling.mult_norm.mbe_id.eval()
    with torch.no_grad():
        y_id = scaling.mult_norm.mbe_id(x_id).squeeze()
    
    print(f"  MSE: {F.mse_loss(y_id, x_id.squeeze()).item():.6f}")

    if args.plot:
        plt.figure(figsize=(8, 6))
        plt.plot(x_id.squeeze().numpy(), x_id.squeeze().numpy(), 'b-', label='True $x$')
        plt.plot(x_id.squeeze().numpy(), y_id.numpy(), 'r--', label='MBE Identity')
        plt.title(f"Internal Identity Mapping [-1, 1] (T={args.timesteps})")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'mbe_ln_step3_id.png'))
        print(f"  Component plots saved to {plot_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MBE LayerNorm Approximation Test")
    parser.add_argument('--test', choices=['invsqrt', 'steps', 'layernorm', 'components', 'all'],
                        default='all', help='Which test to run')
    parser.add_argument('--num_basis', type=int, default=8, help='Number of basis components (N)')
    parser.add_argument('--timesteps', type=int, default=16, help='Number of timesteps (T)')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs per module')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--tv_weight', type=float, default=0.0, help='Weight for TV regularization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Plot comparison results')
    
    args = parser.parse_args()
    set_seed(args.seed)

    if args.test in ('invsqrt', 'all'):
        test_invsqrt(args)
    if args.test in ('steps', 'all'):
        test_step_by_step(args)
    if args.test in ('components', 'all'):
        test_components(args)
    if args.test in ('layernorm', 'all'):
        test_layernorm(args)
    print("\nDone.")
