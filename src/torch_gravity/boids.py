# %%  ──────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import io
import math
import pathlib
import shutil
import tempfile
import warnings
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Video
from matplotlib.patches import FancyArrow

# ────────────────────────────── rendering ────────────────────────────────────
RGB_Int = Tuple[int, int, int]  # 0‑255
RGB_Float = Tuple[float, float, float]  # 0‑1


def _to_float_rgb(c) -> RGB_Float:
    arr = torch.as_tensor(c).detach().cpu().numpy().astype(float)
    if arr.min() < 0:
        raise ValueError("RGB values can’t be negative.")
    return tuple(arr[:3] / (255.0 if arr.max() > 1.0 else 1.0))


def render_frame(
    pos: torch.Tensor,  # [N,2]
    r: Sequence[float] | torch.Tensor,  # len N
    rgb: Sequence[RGB_Int | RGB_Float] | torch.Tensor,
    vel: torch.Tensor | None = None,  # [N,2]
    size: Tuple[int, int] = (800, 800),
    bg: RGB_Int | RGB_Float = (0, 0, 0),
    *,
    window: Tuple[float, float, float, float] | None = None,  # (xmin,xmax,ymin,ymax)
    margin_factor: float = 1.05,
    edge_margin: float = 0.10,
    head_frac: float = 0.5,
    shaft_width_frac: float = 0.25,
    head_width_frac: float = 0.55,
) -> np.ndarray:  # returns H×W×3 uint8
    # normalise ───────────────────────────────────────────────────────────────
    pos_np = pos.detach().cpu().numpy()
    r_np = np.abs(np.asarray(r, dtype=float))
    rgb_np = (
        torch.as_tensor(rgb).detach().cpu().numpy()
        if isinstance(rgb, torch.Tensor)
        else np.asarray(rgb)
    )

    # fig / axes ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.set_facecolor(_to_float_rgb(bg))
    ax.set_aspect("equal")
    ax.axis("off")

    # camera ──────────────────────────────────────────────────────────────────
    if window is None:
        margin = (r_np.max() if r_np.size else 0) * margin_factor
        xmin, xmax = pos_np[:, 0].min() - margin, pos_np[:, 0].max() + margin
        ymin, ymax = pos_np[:, 1].min() - margin, pos_np[:, 1].max() + margin
    else:
        xmin, xmax, ymin, ymax = window
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # boids' bodies ───────────────────────────────────────────────────────────
    for (x, y), rad, col in zip(pos_np, r_np, rgb_np):
        ax.add_patch(plt.Circle((x, y), rad, color=_to_float_rgb(col)))

    # velocity arrows ─────────────────────────────────────────────────────────
    if vel is not None:
        vel_np = vel.detach().cpu().numpy()
        unit_dir = vel_np / (np.linalg.norm(vel_np, axis=1, keepdims=True) + 1e-8)
        for (cx, cy), u, rad in zip(pos_np, unit_dir, r_np):
            diam_len = 2.0 * rad * (1.0 - edge_margin)
            head_len = min(rad * head_frac, diam_len * 0.5)
            shaft_len = diam_len - head_len
            tail = np.array([cx, cy]) - u * (shaft_len * 0.5 + head_len * 0.5)
            dxdy = u * (shaft_len + head_len)
            ax.add_patch(
                FancyArrow(
                    tail[0],
                    tail[1],
                    dxdy[0],
                    dxdy[1],
                    width=rad * shaft_width_frac,
                    head_length=head_len,
                    head_width=rad * head_width_frac,
                    length_includes_head=True,
                    color="white",
                    linewidth=0,
                    alpha=0.9,
                )
            )

    # rasterise ───────────────────────────────────────────────────────────────
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.get_renderer().buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def square_window_from_initial(
    pos0: torch.Tensor, scale: float = 4.0
) -> Tuple[float, float, float, float]:
    """Square window centred on initial COM, enlarged by `scale`× span."""
    xymin = pos0.min(0).values
    xymax = pos0.max(0).values
    centre = 0.5 * (xymin + xymax)
    half = 0.5 * (xymax - xymin).max() * scale
    return (centre[0] - half, centre[0] + half, centre[1] - half, centre[1] + half)


# ────────────────────────────── boids core ───────────────────────────────────
def default_params(**overrides) -> Dict[str, float]:
    p = dict(
        dt=0.025,
        neighbor_dist=1.0,
        separation_dist=0.15,
        cohesion_weight=1.0,
        alignment_weight=1.0,
        separation_weight=1.5,
        max_speed=2.0,
        max_force=0.05,
        world_half_size=2.5,  # toroidal square ±half
    )
    p.update(overrides)
    return p


def step(
    state: Dict[str, torch.Tensor], p: Dict[str, float]
) -> Dict[str, torch.Tensor]:
    """One Euler step of Reynolds boids in a toroidal box."""
    pos, vel = state["pos"], state["vel"]
    dt, vmax, fmax = p["dt"], p["max_speed"], p["max_force"]
    nd, sd = p["neighbor_dist"], p["separation_dist"]

    d = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N,N,2]
    dist = d.norm(dim=-1) + 1e-12

    neigh = (0.0 < dist) & (dist < nd)
    sep_m = (0.0 < dist) & (dist < sd)
    counts = neigh.sum(1, keepdim=True).clamp_(min=1)

    centre = (neigh.float().unsqueeze(-1) * pos.unsqueeze(0)).sum(1) / counts
    coh = (centre - pos) * p["cohesion_weight"]

    mean_v = (neigh.float().unsqueeze(-1) * vel.unsqueeze(0)).sum(1) / counts
    ali = (mean_v - vel) * p["alignment_weight"]

    sep = (-d / dist.unsqueeze(-1)).masked_fill(~sep_m.unsqueeze(-1), 0.0).sum(1)
    sep *= p["separation_weight"]

    acc = coh + ali + sep
    acc_len = acc.norm(dim=1, keepdim=True).clamp(min=1e-12)
    acc = acc / acc_len * acc_len.clamp(max=fmax)

    vel = vel + acc * dt
    speed = vel.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vel = vel / speed * speed.clamp(max=vmax)

    pos = pos + vel * dt
    half = p["world_half_size"]
    pos = (pos + half) % (2 * half) - half

    return {"pos": pos, "vel": vel, "r": state["r"], "rgb": state["rgb"]}


def rollout(
    init: Dict[str, torch.Tensor], p: Dict[str, float], steps: int
) -> List[Dict[str, torch.Tensor]]:
    traj = [init]
    for _ in range(steps):
        traj.append(step(traj[-1], p))
    return traj


# ─────────────────────────── video helpers ───────────────────────────────────
def render_rollout(
    traj: List[Dict[str, torch.Tensor]], size=(500, 500), bg=(0, 0, 0), window=None
) -> List[np.ndarray]:
    frames = []
    for s in traj:
        frames.append(
            render_frame(
                s["pos"],
                s["r"],
                s["rgb"],
                vel=s["vel"],
                size=size,
                bg=bg,
                window=window,
            )
        )
    return frames


def video_from_frames(
    frames: List[np.ndarray], fps: int = 30, path: str | pathlib.Path | None = None
) -> Video:
    path = (
        pathlib.Path(path) if path else pathlib.Path(tempfile.mkdtemp()) / "boids.mp4"
    )
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for f in frames:
        writer.append_data(f)
    writer.close()
    return Video(str(path), embed=True)


# ───────────────────────────── mini‑demo ─────────────────────────────────────
# # ─── Mini‑demo: run after the big block is loaded ────────────────────────────
torch.manual_seed(0)

# 1· initial state -----------------------------------------------------------
N, dev = 40, "cpu"
state0 = {
    "pos": (torch.rand(N, 2) * 5 - 2.5).to(dev),
    "vel": torch.randn(N, 2).to(dev) * 0.4,
    "r": torch.full((N,), 0.12, device=dev),
    "rgb": torch.randint(0, 256, (N, 3), device=dev),
}

# 2· parameters --------------------------------------------------------------
p = default_params(
    neighbor_dist=1.2,
    separation_dist=0.25,
    cohesion_weight=0.5,
    alignment_weight=1.0,
    separation_weight=2.2,
    max_speed=1.8,
    max_force=0.2,  # let them actually turn
    dt=0.03,
)

# 3· simulate ---------------------------------------------------------------
traj = rollout(state0, p, steps=300)

# 4· fixed camera = your toroidal box ----------------------------------------
half = p["world_half_size"]
window = (-half, half, -half, half)  # never changes

# 5· render + encode (use 512×512 ⇒ multiples of 16 → no FFmpeg warning) -----
frames = render_rollout(traj, size=(512, 512), bg=(250, 250, 250), window=window)

video = video_from_frames(frames, fps=30)
video  # Jupyter / IPython will embed and autoplay
