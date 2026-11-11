"""Two-pass exact gradient accumulation for Sharpe-ratio training.

Implements analytically derived gradients for the pooled Sharpe ratio and
optional soft-min per-sample Sharpe, allowing large effective batch sizes
via chunked forward passes without the approximation error of naive
gradient accumulation.
"""

import logging
import math

import torch

from deepm.data.dataset import unpack_torch_dataset
from deepm.models.base import DmnMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch slicing helpers
# ---------------------------------------------------------------------------

def _infer_batch_size(obj: object) -> int | None:
    """Recursively infer batch size from a nested tensor / dict / list."""
    if torch.is_tensor(obj):
        return obj.shape[0]
    if isinstance(obj, dict):
        for v in obj.values():
            b = _infer_batch_size(v)
            if b is not None:
                return b
    if isinstance(obj, (list, tuple)):
        for v in obj:
            b = _infer_batch_size(v)
            if b is not None:
                return b
    return None


def _slice_batch(obj: object, sl: slice) -> object:
    """Recursively slice a nested tensor / dict / list along dim-0."""
    if torch.is_tensor(obj):
        return obj[sl]
    if isinstance(obj, dict):
        return {k: _slice_batch(v, sl) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_slice_batch(v, sl) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# RNG state management (for deterministic replay)
# ---------------------------------------------------------------------------

def _get_rng_state() -> tuple:
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    return (cpu, cuda, mps)


def _set_rng_state(state: tuple) -> None:
    cpu, cuda, mps = state
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    if mps is not None:
        torch.mps.set_rng_state(mps)


# ---------------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def _stream_sum_sumsq(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Accumulate sum and sum-of-squares in float64 for numerical stability."""
    if x.device.type == "mps":  # MPS has no float64
        x = x.cpu()
    xd = x.double()
    return xd.sum(), (xd * xd).sum(), x.numel()


def _pooled_mean_std_pop(
    sum_x: torch.Tensor, sum_x2: torch.Tensor, n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Population mean and std from streaming accumulators."""
    mean = sum_x / n
    var = (sum_x2 / n) - (mean * mean)
    var = torch.clamp(var, min=0.0)
    return mean, torch.sqrt(var)


# ---------------------------------------------------------------------------
# Analytical gradient functions
# ---------------------------------------------------------------------------

def _grad_pooled_loss_wrt_returns_pop(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    n_total: int,
    annual: float = 252.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gradient of ``-pooled_sharpe`` w.r.t. each return element (population std)."""
    a = math.sqrt(annual)
    s = std.clamp_min(1e-12)
    d = (std + eps)
    return -a * (
        (1.0 / n_total) * (1.0 / d)
        - (mean / (d * d)) * ((x - mean) / (n_total * s))
    )


def _grad_per_sample_sharpe_wrt_returns_pop(
    r_bt: torch.Tensor,
    annual: float = 252.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Gradient of per-sample Sharpe w.r.t. each return (population std over time)."""
    a = math.sqrt(annual)
    B, T = r_bt.shape
    mu = r_bt.mean(dim=1, keepdim=True)
    c = r_bt - mu
    var = (c * c).mean(dim=1, keepdim=True)
    std = torch.sqrt(torch.clamp(var, min=0.0)).clamp_min(1e-12)
    d = std + eps

    return a * (
        (1.0 / T) * (1.0 / d)
        - (mu / (d * d)) * (c / (T * std))
    )


# ---------------------------------------------------------------------------
# Main two-pass training step
# ---------------------------------------------------------------------------

def train_step_exact_chunked(
    model,
    optimizer,
    samples,
    train_data,
    device,
    max_gradient_norm: float,
    q: float,
    grad_accum_ratio: int,
):
    """Two-pass gradient-accumulation training step with exact Sharpe gradients.

    Supports three objective variants:
      - pooled-only (neither softmin nor joint): gradient of batch-pooled Sharpe
      - softmin: softmin-weighted per-sample Sharpe
      - joint: pooled + lambda * softmin

    Pass 1 (no_grad): computes pooled statistics (and per-sample Sharpe ratios
    + softmin weights when softmin/joint is active).
    Pass 2 (with grad): replays the same RNG states and backpropagates using
    analytically derived gradients.
    """
    if q != 2.0:
        raise ValueError("Exact chunking requires q=2.0.")

    B_total = _infer_batch_size(samples)
    if B_total is None:
        raise RuntimeError("Could not infer batch size from samples.")

    chunk_size = math.ceil(B_total / grad_accum_ratio)

    use_softmin = getattr(model, "use_softmin_sharpe", False)
    use_joint = getattr(model, "use_joint_pooled_softmin_sharpe", False)
    has_softmin = use_softmin or use_joint

    beta = float(model.softmin_sharpe_beta) if has_softmin else 0.0
    group_size = model.softmin_sharpe_group_size if has_softmin else None
    center = bool(model.softmin_sharpe_center) if has_softmin else False
    lam = float(model.joint_softmin_lambda) if use_joint else 1.0
    eps = float(1e-4)
    annual = 252.0

    # -------------------------
    # PASS 1 (no_grad): pooled stats (+ per-sample SR & softmin weights)
    # -------------------------
    rng_states = []
    acc_device = "cpu" if torch.device(device).type == "mps" else device  # MPS has no float64
    sum_x = torch.zeros((), device=acc_device, dtype=torch.float64)
    sum_x2 = torch.zeros((), device=acc_device, dtype=torch.float64)
    n_total = 0

    sharpe_list = []

    model.train()

    with torch.no_grad():
        for start in range(0, B_total, chunk_size):
            sl = slice(start, min(start + chunk_size, B_total))
            chunk_samples = _slice_batch(samples, sl)

            rng_states.append(_get_rng_state())

            torch_dataset = unpack_torch_dataset(
                chunk_samples, train_data, device, use_dates_mask=False, live_mode=False
            )
            torch_dataset["batch_size"] = sl.stop - sl.start

            returns_bt, _reg = model(
                **torch_dataset,
                mode=DmnMode.TRAINING,
                q=q,
                return_returns_and_reg=True,
            )

            sx, sx2, n_batch = _stream_sum_sumsq(returns_bt)
            sum_x += sx
            sum_x2 += sx2
            n_total += n_batch

            if has_softmin:
                mu_b = returns_bt.mean(dim=1)
                std_b = returns_bt.std(dim=1, correction=0) + eps
                sr_b = (mu_b / std_b) * math.sqrt(annual)
                sharpe_list.append(sr_b.detach())

    mean_g, std_g = _pooled_mean_std_pop(sum_x, sum_x2, n_total)
    mean_g = mean_g.to(device=device, dtype=torch.float32)
    std_g = std_g.to(device=device, dtype=torch.float32)

    loss_pool = -(mean_g / (std_g + eps)) * math.sqrt(annual)

    if has_softmin:
        sharpe_all = torch.cat(sharpe_list, dim=0)
        if group_size is None:
            group_size = sharpe_all.numel()

        if sharpe_all.numel() % group_size != 0:
            raise ValueError(f"Batch {sharpe_all.numel()} not divisible by group_size={group_size}.")

        G = sharpe_all.numel() // group_size
        sharpe_gb = sharpe_all.view(G, group_size)

        w_gb = torch.softmax(-beta * sharpe_gb, dim=1)
        w_all = w_gb.reshape(-1).detach()

        lse = torch.logsumexp(-beta * sharpe_gb, dim=1)
        if center:
            lse = lse - math.log(group_size)
        loss_soft = (lse / beta).mean()

        loss_virtual = (loss_pool + lam * loss_soft) if use_joint else loss_soft
    else:
        w_all = None
        loss_virtual = loss_pool

    train_metric = (-loss_virtual).detach().cpu().item()

    # -------------------------
    # PASS 2 (with grad): replay RNG, backprop via external gradient
    # -------------------------
    optimizer.zero_grad(set_to_none=True)

    offset = 0
    for i, start in enumerate(range(0, B_total, chunk_size)):
        sl = slice(start, min(start + chunk_size, B_total))
        chunk_samples = _slice_batch(samples, sl)

        _set_rng_state(rng_states[i])

        torch_dataset = unpack_torch_dataset(
            chunk_samples, train_data, device, use_dates_mask=False, live_mode=False
        )
        torch_dataset["batch_size"] = sl.stop - sl.start

        returns_bt, reg_loss = model(
            **torch_dataset,
            mode=DmnMode.TRAINING,
            q=q,
            return_returns_and_reg=True,
        )

        if has_softmin and i == 0:
            with torch.no_grad():
                mu_check = returns_bt.mean(dim=1)
                std_check = returns_bt.std(dim=1, correction=0) + eps
                sr_check = (mu_check / std_check) * math.sqrt(annual)
                diff = (sr_check - sharpe_list[i]).abs().max().item()
                if diff > 1e-5:
                    logger.warning("Pass 1 vs Pass 2 mismatch! Max Diff: %.2e" % diff)
                    logger.warning("Dropout mask or data order is not aligned between passes.")

        g_total = _grad_pooled_loss_wrt_returns_pop(
            x=returns_bt,
            mean=mean_g.to(returns_bt.dtype),
            std=std_g.to(returns_bt.dtype),
            n_total=n_total,
            annual=annual,
            eps=eps,
        )

        if has_softmin:
            b = returns_bt.shape[0]
            w = w_all[offset:offset + b].to(device=returns_bt.device, dtype=returns_bt.dtype)
            offset += b

            dSR_dr = _grad_per_sample_sharpe_wrt_returns_pop(
                r_bt=returns_bt,
                annual=annual,
                eps=eps,
            )
            g_soft = (-(lam / G) * w).unsqueeze(1) * dSR_dr
            g_total = g_total + g_soft

        if not torch.is_tensor(reg_loss):
            reg_loss = torch.zeros((), device=returns_bt.device, dtype=returns_bt.dtype)

        g_total = g_total.detach()

        surrogate = (returns_bt * g_total).sum()

        if torch.is_tensor(reg_loss) and (reg_loss.requires_grad or reg_loss.grad_fn is not None):
            surrogate = surrogate + reg_loss

        surrogate.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
    optimizer.step()
    return train_metric
