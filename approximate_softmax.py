"""
MBE Neuron-based Softmax Approximation
=======================================
Softmax(x_i) = e^{x_i} / sum_j(e^{x_j}) 를 세 단계로 분해하여 근사:

  1. 지수 함수 근사  : e^{x_i} = 2^{floor(x*log2e)} * 2^{frac(x*log2e)}
                      정수부 → shift 연산, 소수부 → MBE 뉴런
  2. 역수 근사       : 1/sum_j(e^{x_j}) = (1/M) * 2^{-E}
                      IEEE754 분해 → 가수 역수는 MBE 뉴런, 지수부 → shift 연산
  3. FP 곱셈 결합   : Softmax = e^{x_i} * (1/sum_j e^{x_j})
                      MBEMultiplier (Spike Matrix × Intensity Matrix) 로 근사
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
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

# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────────────────────────────────────
LOG2E = math.log2(math.e)   # log_2(e) ≈ 1.44269...

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'mbe_models', 'softmax')


def _load_or_train(model_path: str, target_func, x_range, tag: str,
                   num_basis=8, timesteps=16, num_epochs=2000, lr=0.01, tv_weight=0.0) -> MBENeuron:
    """저장된 모델이 있으면 로드, 없으면 학습 후 저장."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if os.path.exists(model_path):
        print(f"[{tag}] Loading pre-trained model from {model_path}")
        return MBENeuron.load(model_path)

    print(f"[{tag}] Training MBE neuron on range {x_range} ...")
    x_probe = torch.linspace(x_range[0], x_range[1], 4000)
    alpha = float(target_func(x_probe).abs().max().item())
    alpha = max(alpha, 1e-3)

    model = train_mbe_neuron(
        target_func=target_func,
        x_range=x_range,
        num_samples=10000,
        num_epochs=num_epochs,
        lr=lr,
        tv_weight=tv_weight,
        num_basis=num_basis,
        timesteps=timesteps,
        alpha=alpha,
    )
    model.save(model_path)
    print(f"[{tag}] Saved to {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – 지수 함수 근사
# ─────────────────────────────────────────────────────────────────────────────
class MBEExp(nn.Module):
    """
    e^x 를 MBE 뉴런으로 근사.

    분해:
        y   = x * log2(e)
        E_i = floor(y)          # 정수부 → shift (× 2^E_i)
        f   = y - E_i           # 소수부 ∈ [0,1) → MBE 뉴런으로 2^f 근사

    최종: e^x ≈ MBE(f) * 2^{E_i}
    """

    # softmax 전용 모델명 (기존 approximate_activation.py 모델과 충돌 방지)
    DEFAULT_MODEL = 'mbe_softmax_exp_frac.pth'

    def __init__(self, num_basis: int = 8, timesteps: int = 16,
                 model_path: str = None, epochs=3000, lr=0.005, tv_weight=0.05):
        super().__init__()
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, self.DEFAULT_MODEL)

        def frac_exp(x): return torch.pow(2.0, x)

        self.mbe_frac = _load_or_train(
            model_path, frac_exp,
            x_range=(0.0, 1.0 - 1e-4),
            tag='MBEExp-frac',
            num_basis=num_basis,
            timesteps=timesteps,
            num_epochs=epochs,
            lr=lr,
            tv_weight=tv_weight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: 임의 shape 의 실수 텐서
        반환: e^x 근사값 (같은 shape)
        """
        y = x * LOG2E                       # x·log₂e
        E = torch.floor(y)                  # 정수부
        frac = y - E                        # 소수부 ∈ [0, 1)

        # MBE로 2^frac 근사 – 뉴런 입력은 (batch, 1) 형태를 기대
        orig_shape = frac.shape
        frac_flat = frac.reshape(-1, 1)

        self.mbe_frac.eval()
        with torch.no_grad():
            frac_approx = self.mbe_frac(frac_flat).reshape(orig_shape)

        # 정수부 shift: 2^E (float 도메인에서 곱셈)
        int_part = torch.pow(2.0, E)

        return int_part * frac_approx


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – 역수 근사
# ─────────────────────────────────────────────────────────────────────────────
def _fp32_decompose(v: torch.Tensor):
    """
    스칼라 텐서 v (> 0) 를 IEEE 754 스타일로 분해.
        v = M * 2^E,  M ∈ [1, 2)

    반환: (M, E)  – float Tensor
    """
    # frexp: mantissa ∈ [0.5, 1.0), exponent 반환 → M·2^E, M ∈ [0.5,1)
    # 논문 표기와 맞추기 위해 M ∈ [1,2) 로 변환
    E_raw = torch.floor(torch.log2(v.abs().clamp(min=1e-38)))
    M = v / torch.pow(2.0, E_raw)          # M ∈ [1, 2)
    return M, E_raw


class MBEReciprocal(nn.Module):
    """
    양수 스칼라 S = M·2^E 의 역수를 MBE 뉴런으로 근사.

    분해:
        1/S = (1/M) * 2^{-E}
        1/M: M ∈ [1,2) → MBE 뉴런이 1/M 근사
        2^{-E}: 정수 지수 → shift 연산

    반환: 1/S 근사값 (스칼라 Tensor)
    """

    # softmax 전용 모델명
    DEFAULT_MODEL = 'mbe_softmax_recip.pth'

    def __init__(self, num_basis: int = 8, timesteps: int = 16,
                 model_path: str = None, epochs=3000, lr=0.005, tv_weight=0.05):
        super().__init__()
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, self.DEFAULT_MODEL)

        def recip_mantissa(x): return 1.0 / x.clamp(min=1e-6)

        self.mbe_recip = _load_or_train(
            model_path, recip_mantissa,
            x_range=(1.0, 2.0 - 1e-4),
            tag='MBEReciprocal',
            num_basis=num_basis,
            timesteps=timesteps,
            num_epochs=epochs,
            lr=lr,
            tv_weight=tv_weight
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: 양수 스칼라(또는 임의 shape) Tensor
        반환: 1/s 근사값
        """
        M, E = _fp32_decompose(s)

        orig_shape = M.shape
        M_flat = M.reshape(-1, 1)

        self.mbe_recip.eval()
        with torch.no_grad():
            recip_M = self.mbe_recip(M_flat).reshape(orig_shape)  # ≈ 1/M

        # 지수 역수: 2^{-E}
        recip_shift = torch.pow(2.0, -E)

        return recip_M * recip_shift


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – MBE Softmax
# ─────────────────────────────────────────────────────────────────────────────
class MBESoftmax(nn.Module):
    """
    MBE 뉴런 기반 Softmax 근사.

    Softmax(x_i) = e^{x_i} * (1 / sum_j e^{x_j})

    파이프라인:
        1. MBEExp      : 각 x_i → e^{x_i} 근사
        2. MBEReciprocal: sum_j e^{x_j} → 1/sum 근사
        3. MBEMultiplier: e^{x_i} * (1/sum) → 스파이크 기반 FP 곱셈 근사

    Args:
        dim          : Softmax 를 적용할 차원 (default: -1)
        num_basis    : MBE 뉴런의 기저 컴포넌트 수
        timesteps    : 스파이크 시뮬레이션 타임스텝
        use_mbe_mult : True → MBEMultiplier 사용, False → 일반 곱셈 (디버깅용)
        model_dir    : 사전학습 모델 저장/로드 디렉터리
    """

    def __init__(
        self,
        dim: int = -1,
        num_basis: int = 8,
        timesteps: int = 16,
        use_mbe_mult: bool = True,
        model_dir: str = MODEL_DIR,
        epochs: int = 3000,
        lr: float = 0.005,
        tv_weight: float = 0.05,
    ):
        super().__init__()
        self.dim = dim
        self.use_mbe_mult = use_mbe_mult

        # --- Step 1: 지수 함수 MBE (softmax 전용 모델) ---
        self.mbe_exp = MBEExp(
            num_basis=num_basis,
            timesteps=timesteps,
            model_path=os.path.join(model_dir, MBEExp.DEFAULT_MODEL),
            epochs=epochs, lr=lr, tv_weight=tv_weight
        )

        # --- Step 2: 역수 MBE (softmax 전용 모델) ---
        self.mbe_recip = MBEReciprocal(
            num_basis=num_basis,
            timesteps=timesteps,
            model_path=os.path.join(model_dir, MBEReciprocal.DEFAULT_MODEL),
            epochs=epochs, lr=lr, tv_weight=tv_weight
        )

        # --- Step 3: FP 곱셈 MBE ---
        if use_mbe_mult:
            id_path = os.path.join(model_dir, 'mbe_softmax_id.pth')
            def identity_func(x): return x

            mbe_id = _load_or_train(
                id_path, identity_func,
                x_range=(0.0, 1.0),
                tag='MBEId-softmax',
                num_basis=num_basis,
                timesteps=timesteps,
                num_epochs=epochs,
                lr=lr,
                tv_weight=tv_weight
            )
            self.mbe_mult = MBEMultiplier(mbe_id_model=mbe_id)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) shape Tensor
        반환: Softmax 근사값, 같은 shape
        """
        # ── 수치 안정성: x - max(x) (shift 는 softmax 값에 영향 없음) ──
        # 결과적으로 exp_x 의 최대값은 1.0, 나머지는 (0, 1] → MBEId [0,1] 범위 보장
        x_stable = x - x.max(dim=self.dim, keepdim=True).values

        # ── Step 1: e^{x_i} 근사 ─────────────────────────────────────
        exp_x = self.mbe_exp(x_stable)           # shape: (..., D)
        exp_x = exp_x.clamp(min=0.0)             # 근사 오차로 인한 음수 방지

        # ── Step 2: sum & 역수 근사 ──────────────────────────────────
        sum_exp = exp_x.sum(dim=self.dim, keepdim=True)   # shape: (..., 1)
        recip_sum = self.mbe_recip(sum_exp)               # shape: (..., 1)
        recip_sum = recip_sum.clamp(min=0.0)              # 안전 클리핑

        # ── Step 3: FP 곱셈 ──────────────────────────────────────────
        if self.use_mbe_mult:
            # MBEMultiplier: 피연산자가 [0,1] 범위임을 보장 (stable trick 덕분)
            recip_broadcast = recip_sum.expand_as(exp_x)
            out = self.mbe_mult(exp_x, recip_broadcast)
            out = out.clamp(min=0.0)   # 음수 방지
        else:
            # 디버깅용 단순 곱셈 (MBEMultiplier 없이 exp+recip 만 검증)
            out = exp_x * recip_sum
            out = out.clamp(min=0.0)

        # ── 재정규화: exp/recip/mult 오차 누적으로 합이 1 에서 벗어나는 것을 보정 ──
        # Softmax 는 확률분포 → 합=1 이 수학적 조건.
        # 각 단계의 근사 오차가 곱해지면서 row-sum 이 1 에서 벗어나므로
        # 최종적으로 row-sum 으로 나누어 정규화. 상대적 크기(순위)는 보존됨.
        out_sum = out.sum(dim=self.dim, keepdim=True).clamp(min=1e-8)
        out = out / out_sum

        return out


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 & 데모
# ─────────────────────────────────────────────────────────────────────────────
def _print_row(label, values):
    vals = " | ".join(f"{v:8.5f}" for v in values)
    print(f"  {label:20s}: {vals}")


def test_mbe_exp(args=None):
    print("\n" + "=" * 60)
    print("  [TEST] MBEExp — 지수 함수 근사")
    print("=" * 60)
    num_basis = args.num_basis if args else 8
    timesteps = args.timesteps if args else 16
    mbe_exp = MBEExp(num_basis=num_basis, timesteps=timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)
    
    x = torch.linspace(-2.0, 4.0, 100)
    true_val = torch.exp(x)
    approx_val = mbe_exp(x)
    
    _print_row("x (sample)",     x[::20].tolist())
    _print_row("e^x (true)",    true_val[::20].tolist())
    _print_row("MBE approx",    approx_val[::20].tolist())
    
    if args and args.plot:
        plot_dir = os.path.join('plots', 'softmax')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(x.numpy(), true_val.numpy(), 'b-', label='True $e^x$')
        plt.plot(x.numpy(), approx_val.numpy(), 'r--', label='MBE Approx')
        plt.title(f"MBE Exponential Approximation (T={timesteps})")
        plt.xlabel("x")
        plt.ylabel("$e^x$")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(plot_dir, 'mbe_softmax_step1_exp.png')
        plt.savefig(save_path)
        print(f"  Plot saved to {save_path}")


def test_mbe_reciprocal(args=None):
    print("\n" + "=" * 60)
    print("  [TEST] MBEReciprocal — 역수 근사")
    print("=" * 60)
    num_basis = args.num_basis if args else 8
    timesteps = args.timesteps if args else 16
    mbe_recip = MBEReciprocal(num_basis=num_basis, timesteps=timesteps, epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight)
    
    s = torch.linspace(0.5, 10.0, 100)
    true_val = 1.0 / s
    approx_val = mbe_recip(s)
    
    _print_row("S (sample)",     s[::20].tolist())
    _print_row("1/S (true)",    true_val[::20].tolist())
    _print_row("MBE approx",    approx_val[::20].tolist())

    if args and args.plot:
        plot_dir = os.path.join('plots', 'softmax')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(s.numpy(), true_val.numpy(), 'b-', label='True $1/S$')
        plt.plot(s.numpy(), approx_val.numpy(), 'r--', label='MBE Approx')
        plt.title(f"MBE Reciprocal Approximation (T={timesteps})")
        plt.xlabel("S")
        plt.ylabel("1/S")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(plot_dir, 'mbe_softmax_step2_recip.png')
        plt.savefig(save_path)
        print(f"  Plot saved to {save_path}")

def test_mbe_id(args):
    print("\n" + "=" * 60)
    print("  [TEST] MBE Identity — 곱셈용 항등 함수 근사")
    print("=" * 60)
    # Softmax 내부적으로 [0,1] 범위를 사용하므로 동일하게 테스트
    id_path = os.path.join(MODEL_DIR, 'mbe_softmax_id.pth')
    def identity_func(x): return x
    
    mbe_id = _load_or_train(
        id_path, identity_func, x_range=(0.0, 1.0), tag='MBEId-softmax',
        num_basis=args.num_basis, timesteps=args.timesteps,
        num_epochs=args.epochs, lr=args.lr, tv_weight=args.tv_weight
    )
    
    x = torch.linspace(0.0, 1.0, 100).unsqueeze(1)
    mbe_id.eval()
    with torch.no_grad():
        approx_val = mbe_id(x).squeeze()
    
    if args.plot:
        plot_dir = os.path.join('plots', 'softmax')
        os.makedirs(plot_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.plot(x.squeeze().numpy(), x.squeeze().numpy(), 'b-', label='True $x$')
        plt.plot(x.squeeze().numpy(), approx_val.numpy(), 'r--', label='MBE Approx')
        plt.title(f"MBE Identity Mapping [0, 1] (T={args.timesteps})")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(plot_dir, 'mbe_softmax_step3_id.png')
        plt.savefig(save_path)
        print(f"  Plot saved to {save_path}")


def test_mbe_softmax(args, use_mbe_mult: bool = True):
    tag = "MBEMultiplier" if use_mbe_mult else "Plain multiply"
    print("\n" + "=" * 60)
    print(f"  [TEST] MBESoftmax ({tag})")
    print("=" * 60)

    # (2, 4) batch 테스트
    x = torch.tensor([
        [2.0, 1.0, 0.1, -1.0],
        [0.5, 0.5, 0.5,  0.5],
    ])

    true_sm  = F.softmax(x, dim=-1)
    model    = MBESoftmax(
        dim=-1, 
        num_basis=args.num_basis, 
        timesteps=args.timesteps, 
        use_mbe_mult=use_mbe_mult,
        epochs=args.epochs,
        lr=args.lr,
        tv_weight=args.tv_weight
    )
    model.eval()
    with torch.no_grad():
        approx_sm = model(x)

    print("  Input x:")
    for row in x:
        print("   ", [f"{v:.3f}" for v in row.tolist()])
    print("  True Softmax:")
    for row in true_sm:
        print("   ", [f"{v:.5f}" for v in row.tolist()])
    print("  MBE Softmax Approx:")
    for row in approx_sm:
        print("   ", [f"{v:.5f}" for v in row.tolist()])

    mse = F.mse_loss(approx_sm, true_sm).item()
    print(f"  MSE: {mse:.8f}")
    # 합이 1 에 가까운지 확인
    row_sums = approx_sm.sum(dim=-1)
    print(f"  Row sums (should ≈ 1): {row_sums.tolist()}")

    if getattr(args, 'plot', False):
        plot_dir = os.path.join('plots', 'softmax')
        os.makedirs(plot_dir, exist_ok=True)
        num_rows = x.shape[0]
        fig, axes = plt.subplots(num_rows, 1, figsize=(12, 6 * num_rows))
        if num_rows == 1: axes = [axes]
        
        indices = np.arange(x.shape[1])
        width = 0.35

        for i in range(num_rows):
            t_vals = true_sm[i].detach().numpy()
            a_vals = approx_sm[i].detach().numpy()
            diffs  = a_vals - t_vals

            rects1 = axes[i].bar(indices - width/2, t_vals, width, label='True Softmax', color='#3498db', alpha=0.8)
            rects2 = axes[i].bar(indices + width/2, a_vals, width, label='MBE Softmax', color='#e74c3c', alpha=0.8)
            
            # 수치 텍스트 추가
            for j in range(len(indices)):
                # True 값 표시 (막대 위)
                axes[i].text(indices[j] - width/2, t_vals[j] + 0.01, f'{t_vals[j]:.4f}', 
                             ha='center', va='bottom', fontsize=9, color='#2980b9', fontweight='bold')
                # Approx 값 표시 (막대 위)
                axes[i].text(indices[j] + width/2, a_vals[j] + 0.01, f'{a_vals[j]:.4f}', 
                             ha='center', va='bottom', fontsize=9, color='#c0392b', fontweight='bold')
                # 오차(Error) 표시 (두 막대 사이 위쪽)
                axes[i].text(indices[j], max(t_vals[j], a_vals[j]) + 0.05, f'err:{abs(diffs[j]):.4f}', 
                             ha='center', va='bottom', fontsize=8, color='black', fontstyle='italic',
                             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

            axes[i].set_title(f"Softmax Comparison (Batch {i+1}) - MSE: {F.mse_loss(approx_sm[i], true_sm[i]):.6f}", fontsize=12)
            axes[i].set_xticks(indices)
            axes[i].set_xticklabels([f'x={v:.1f}' for v in x[i].tolist()])
            axes[i].set_ylabel("Probability")
            axes[i].set_ylim(0, max(t_vals.max(), a_vals.max()) + 0.15) # 텍스트 공간 확보
            axes[i].legend()
            axes[i].grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"mbe_softmax_{tag.lower().replace(' ', '_')}.png")
        plt.savefig(save_path)
        print(f"  Visual plot saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MBE Softmax Approximation Test")
    parser.add_argument('--test', choices=['exp', 'recip', 'id', 'softmax', 'all'],
                        default='all', help='Which sub-test to run')
    parser.add_argument('--no_mbe_mult', action='store_true',
                        help='Use plain float multiply instead of MBEMultiplier (faster debug)')
    parser.add_argument('--num_basis', type=int, default=8, help='Number of basis components (N)')
    parser.add_argument('--timesteps', type=int, default=16, help='Number of timesteps (T)')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs per module')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--tv_weight', type=float, default=0.0, help='Weight for TV regularization')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Plot the approximation result')
    
    args = parser.parse_args()
    set_seed(args.seed)

    use_mbe_mult = not args.no_mbe_mult

    if args.test in ('exp', 'all'):
        test_mbe_exp(args)
    if args.test in ('recip', 'all'):
        test_mbe_reciprocal(args)
    if args.test in ('id', 'all'):
        test_mbe_id(args)
    if args.test in ('softmax', 'all'):
        test_mbe_softmax(args, use_mbe_mult=use_mbe_mult)

    print("\nDone.")
