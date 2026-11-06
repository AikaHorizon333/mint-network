"""
MultiScaleCNN v4 — Improved Architecture + Training
=====================================================
Addresses the key bottlenecks identified in the ImprovedCNN audit.

Architecture changes (vs src/rmcls/cnn_improved_v2.py):
  1. Multi-scale attention pooling  (replaces global average pool)
       — taps features after each dilation cycle independently
       — learned attention weights select important time steps
       — preserves information at different temporal scales
  2. Conv-based frequency encoder  (replaces single Linear layer)
       — strided Conv1d on log-magnitude spectrum
       — extracts local spectral patterns hierarchically
       — 30x fewer parameters than the original Linear(10473, 128)
  3. Deeper aggregation head  (3 FC layers, was 2)
       — 256 → 512 → 512 → output_dim
       — richer nonlinear mapping from features to filter taps
  4. GELU activations  (replaces ReLU throughout)
       — smoother gradient landscape, avoids dead neurons
  5. Full T-tap output  (no symmetry constraint by default)
       — allows mixed-phase filters (paper showed linear phase
         emerges naturally even without the constraint)
       — set SYMMETRIC = True to restore half-tap + mirror
  6. Near-zero init on output layer
       — initial filter starts near zero, preventing large-loss spikes

Training changes:
  1. Linear LR warmup  (1000 iters from lr/1000 → lr)
  2. AdamW + ReduceLROnPlateau + gradient clipping  (kept from v2)
  3. Improved spectral loss with log-freq + notch weighting
"""


# ============================================================
# CELL 1 — Imports
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# CELL 2 — Load RIR
# ============================================================

rir_path = "/content/RIR.wav"

rate, data = wavfile.read(rir_path)
fs   = rate
data = data.astype(np.float32)
if data.ndim > 1:
    data = data[:, 0]
data = data / (np.max(np.abs(data)) + 1e-12)

L    = len(data)
h_np = data.copy()
h    = torch.tensor(data, dtype=torch.float32, device=device)

print(f"RIR loaded: fs={fs} Hz, L={L} samples, duration={L/fs:.3f} s")


# ============================================================
# CELL 3 — Filter Hyperparameters
# ============================================================

T         = 1025        # inverse filter length (odd)
SYMMETRIC = False       # True  → output half taps + mirror (linear phase)
                        # False → output all T taps (mixed phase allowed)

half    = (T + 1) // 2
out_dim = half if SYMMETRIC else T

print(f"Filter length T={T},  SYMMETRIC={SYMMETRIC},  output_dim={out_dim}")

def symmetrize(half_taps: torch.Tensor) -> torch.Tensor:
    return torch.cat([half_taps, half_taps[:-1].flip(0)], dim=0)


# ============================================================
# CELL 4 — Improved Spectral Loss Setup
# ============================================================

w       = 100.0
freq_lo = 20
freq_hi = fs // 2

N_fft = L + T - 1
freqs = np.fft.rfftfreq(N_fft, d=1.0 / fs)

freq_mask = torch.tensor(
    (freqs >= freq_lo) & (freqs <= freq_hi),
    dtype=torch.bool, device=device
)

H_mag_np = np.abs(np.fft.rfft(h_np, n=N_fft))

log_weight   = 1.0 / (freqs + 1e-6)
eps_notch    = 1e-3 * H_mag_np.max()
notch_weight = 1.0 / (H_mag_np + eps_notch)

combined = log_weight * notch_weight
combined[~((freqs >= freq_lo) & (freqs <= freq_hi))] = 0.0

n_active = freq_mask.sum().item()
combined = combined / (combined.sum() + 1e-12) * n_active

freq_weight = torch.tensor(combined, dtype=torch.float32, device=device)

print(f"Loss freq range: {freq_lo}–{freq_hi} Hz  ({n_active} active bins)")
print(f"Notch eps: {eps_notch:.4e}  |  Log+Notch weights applied")


# ============================================================
# CELL 5 — Improved Spectral Loss Function
# J = w * sum_f [ freq_weight_f * ( |H(f)*G(f)| - 1 )^2 ]
# ============================================================

def improved_spectral_loss(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    H   = torch.fft.rfft(h, n=N_fft)
    G   = torch.fft.rfft(g, n=N_fft)
    mag = torch.abs(H * G)
    err = mag - 1.0
    return w * torch.sum(freq_weight * err ** 2)


# ============================================================
# CELL 6 — Building Blocks
# ============================================================

class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel,
            dilation=dilation,
            padding=dilation * (kernel - 1) // 2,
        )
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.gelu(self.norm(self.conv(x)))


class AttentionPool(nn.Module):
    """Learned weighted average over the time axis."""
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits  = self.query(x.transpose(1, 2))        # [B, T, 1]
        weights = F.softmax(logits, dim=1)              # [B, T, 1]
        return (x * weights.transpose(1, 2)).sum(dim=-1)  # [B, C]


class FreqEncoder(nn.Module):
    """Strided Conv1d encoder on the log-magnitude spectrum."""
    def __init__(self, freq_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,  32,       kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64,       kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, freq_dim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, log_mag: torch.Tensor) -> torch.Tensor:
        x = log_mag.unsqueeze(1)       # [B, 1, FREQ_IN]
        return self.net(x).squeeze(-1)  # [B, freq_dim]


# ============================================================
# CELL 7 — MultiScaleCNN Architecture
#
#  Time branch:
#    h → Conv1d(1, CH, 1) → GELU
#    → Cycle 1 (10 dilated res blocks) → AttentionPool → t1 [CH]
#    → Cycle 2 (10 dilated res blocks) → AttentionPool → t2 [CH]
#
#  Freq branch:
#    rfft(h) → log|·| → FreqEncoder (strided convs) → f [FREQ_DIM]
#
#  Head:
#    cat(t1, t2, f) → FC(256,512) → GELU
#                    → FC(512,512) → GELU
#                    → FC(512, out_dim)
# ============================================================

TIME_DIM = 64
FREQ_DIM = 128

DILATIONS_PER_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


class MultiScaleCNN(nn.Module):
    def __init__(self, L: int, out_dim: int):
        super().__init__()
        self.L = L

        # --- Time branch ---
        self.input_proj = nn.Conv1d(1, TIME_DIM, kernel_size=1)
        self.cycle1 = nn.Sequential(*[
            DilatedResBlock(TIME_DIM, d) for d in DILATIONS_PER_CYCLE
        ])
        self.cycle2 = nn.Sequential(*[
            DilatedResBlock(TIME_DIM, d) for d in DILATIONS_PER_CYCLE
        ])
        self.attn1 = AttentionPool(TIME_DIM)
        self.attn2 = AttentionPool(TIME_DIM)

        # --- Frequency branch ---
        self.freq_enc = FreqEncoder(freq_dim=FREQ_DIM)

        # --- Aggregation head ---
        agg_in = TIME_DIM * 2 + FREQ_DIM
        self.head = nn.Sequential(
            nn.Linear(agg_in, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

        nn.init.uniform_(self.head[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # --- Time branch ---
        x = h.view(1, 1, -1)
        x = F.gelu(self.input_proj(x))
        x = self.cycle1(x)
        t1 = self.attn1(x)                # [1, TIME_DIM]
        x = self.cycle2(x)
        t2 = self.attn2(x)                # [1, TIME_DIM]

        # --- Frequency branch ---
        H_logmag = torch.log(torch.abs(torch.fft.rfft(h, n=self.L)) + 1e-8)
        f = self.freq_enc(H_logmag.unsqueeze(0))  # [1, FREQ_DIM]

        # --- Aggregate ---
        z = torch.cat([t1, t2, f], dim=-1)
        return self.head(z).squeeze(0)     # [out_dim]


model = MultiScaleCNN(L=L, out_dim=out_dim).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# CELL 8 — Optimizer Setup
# ============================================================

lr           = 1e-3
weight_decay = 1e-4
iters        = 300_000
log_every    = 500
warmup_iters = 1000

checkpoint_dir = "/content/checkpoints_v4"
os.makedirs(checkpoint_dir, exist_ok=True)

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode     = "min",
    factor   = 0.5,
    patience = 8000,
    min_lr   = 1e-6,
)

best_loss = float("inf")
best_g    = None
last_lr   = lr


# ============================================================
# CELL 9 — Training
# ============================================================

model.train()
for t in range(1, iters + 1):

    if t <= warmup_iters:
        warmup_lr = lr * t / warmup_iters
        for pg in opt.param_groups:
            pg["lr"] = warmup_lr
    else:
        scheduler.step(best_loss)

    opt.zero_grad()

    taps = model(h)
    g    = symmetrize(taps) if SYMMETRIC else taps
    loss = improved_spectral_loss(h, g)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()

    lval   = float(loss.detach().cpu())
    cur_lr = opt.param_groups[0]["lr"]

    if lval < best_loss:
        best_loss = lval
        best_g    = g.detach().clone()
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        torch.save(best_g,             os.path.join(checkpoint_dir, "best_g.pt"))

    if t % log_every == 0:
        lr_tag = "  *** LR reduced ***" if cur_lr < last_lr else ""
        print(f"iter={t:>7d}  lr={cur_lr:.2e}  loss={lval:.6e}  best={best_loss:.6e}{lr_tag}")
        last_lr = cur_lr

    if t % 10_000 == 0:
        torch.save({
            'iteration':            t,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss':            best_loss,
            'best_g':               best_g.cpu() if best_g is not None else None,
        }, os.path.join(checkpoint_dir, f"checkpoint_{t:07d}.pth"))

print("Done. best_loss:", best_loss)


# ============================================================
# CELL 10 — Inspect Best Filter
# ============================================================

import matplotlib.pyplot as plt

g_np = best_g.cpu().numpy()
eir  = np.convolve(h_np, g_np)[:L]

print(f"Filter norm: {np.linalg.norm(g_np):.4f}")
print(f"Filter max:  {np.abs(g_np).max():.4f}")
print(f"EIR peak:    {np.abs(eir).max():.4f}  at sample {np.argmax(np.abs(eir))}")

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(g_np)
axes[0].set_title("Best Inverse Filter (g)  —  MultiScaleCNN v4")
axes[0].set_xlabel("Sample"); axes[0].set_ylabel("Amplitude"); axes[0].grid(True)

axes[1].plot(eir)
axes[1].set_title("Equalized IR  (H * g)")
axes[1].set_xlabel("Sample"); axes[1].set_ylabel("Amplitude"); axes[1].grid(True)

eir_mag_db = 20 * np.log10(np.abs(np.fft.rfft(eir, n=N_fft)) + 1e-12)
axes[2].semilogx(freqs, eir_mag_db)
axes[2].axhline(0, color="r", linestyle="--", label="flat target (0 dB)")
axes[2].set_xlim([freq_lo, freq_hi])
axes[2].set_title("Equalized IR — Magnitude Spectrum (dB)")
axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("dB")
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# CELL 11 — Per-band MSE breakdown
# ============================================================

bands = [
    ("Sub-bass",  20,    80),
    ("Bass",      80,    300),
    ("Midrange",  300,   2000),
    ("Presence",  2000,  6000),
    ("Treble",    6000,  freq_hi),
]

H_mag_full = np.abs(np.fft.rfft(h_np, n=N_fft))
G_mag_full = np.abs(np.fft.rfft(g_np, n=N_fft))

print("\nPer-band equalization MSE:")
print(f"{'Band':<12} {'Range':>14}   MSE")
print("-" * 40)
for name, lo, hi in bands:
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() == 0:
        continue
    mse = np.mean((H_mag_full[mask] * G_mag_full[mask] - 1.0) ** 2)
    print(f"{name:<12} {lo:>6}–{hi:<6} Hz   {mse:.4e}")


# ============================================================
# CELL 12 — Save Results as WAV
# ============================================================

def save_results(g: torch.Tensor, tag: str = "best"):
    g_np  = g.cpu().numpy()
    eir   = np.convolve(h_np, g_np)[:L]

    g_norm   = g_np / (np.max(np.abs(g_np))  + 1e-12)
    eir_norm = eir  / (np.max(np.abs(eir))   + 1e-12)

    wavfile.write(os.path.join(checkpoint_dir, f"{tag}_W.wav"),
                  fs, g_norm.astype(np.float32))
    wavfile.write(os.path.join(checkpoint_dir, f"{tag}_EIR.wav"),
                  fs, eir_norm.astype(np.float32))

    H_mag = np.abs(np.fft.rfft(h_np, n=N_fft))
    G_mag = np.abs(np.fft.rfft(g_np, n=N_fft))
    mask  = (freqs >= freq_lo) & (freqs <= freq_hi)
    mse   = np.mean((H_mag[mask] * G_mag[mask] - 1.0) ** 2)
    print(f"MSE (spectral, paper metric): {mse:.6e}")
    print(f"Saved: {tag}_W.wav, {tag}_EIR.wav  →  {checkpoint_dir}/")

save_results(best_g, tag="best")
