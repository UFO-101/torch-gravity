# %%
from __future__ import annotations

from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrow

RGB_Int = Tuple[int, int, int]  # 0‑255
RGB_Float = Tuple[float, float, float]  # 0‑1


# ------------- helper ------------------------------------------------------
def _to_float_rgb(c) -> RGB_Float:
    arr = torch.as_tensor(c).detach().cpu().numpy().astype(float)
    if arr.min() < 0:
        raise ValueError("RGB values can’t be negative.")
    return tuple(arr[:3] / (255.0 if arr.max() > 1.0 else 1.0))


# ------------- renderer ----------------------------------------------------
def render_frame(
    pos: torch.Tensor,  # [N,2]
    r: Sequence[float] | torch.Tensor,  # len N radii
    rgb: Sequence[RGB_Int | RGB_Float] | torch.Tensor,  # len N colours
    vel: torch.Tensor | None = None,  # [N,2] or None
    size: Tuple[int, int] = (800, 800),  # px
    bg: RGB_Int | RGB_Float = (0, 0, 0),
    margin_factor: float = 1.05,  # padding outside swarm bbox (for camera)
    edge_margin: float = 0.10,  # ★ fraction of radius left clear at tail & tip
    head_frac: float = 0.5,  # ★ arrow‑head length as fraction of radius
    shaft_width_frac: float = 0.25,  # ★ shaft thickness relative to radius
    head_width_frac: float = 0.55,  # ★ arrow‑head width   relative to radius
) -> np.ndarray:  # returns H×W×3 uint8
    # ---- normalise arrays ----
    pos_np = pos.detach().cpu().numpy()
    r_np = np.abs(np.asarray(r, dtype=float))
    rgb_np = (
        torch.as_tensor(rgb).detach().cpu().numpy()
        if isinstance(rgb, torch.Tensor)
        else np.asarray(rgb)
    )

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(size[0] / 100, size[1] / 100), dpi=100)
    ax.set_facecolor(_to_float_rgb(bg))
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- auto window ----
    margin = (r_np.max() if r_np.size else 0) * margin_factor
    xmin, xmax = pos_np[:, 0].min() - margin, pos_np[:, 0].max() + margin
    ymin, ymax = pos_np[:, 1].min() - margin, pos_np[:, 1].max() + margin
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ---- draw circles ----
    for (x, y), rad, col in zip(pos_np, r_np, rgb_np):
        ax.add_patch(plt.Circle((x, y), rad, color=_to_float_rgb(col)))

    # ---- draw arrows ----
    if vel is not None:
        vel_np = vel.detach().cpu().numpy()
        unit_dir = vel_np / (np.linalg.norm(vel_np, axis=1, keepdims=True) + 1e-8)

        for (cx, cy), u, rad in zip(pos_np, unit_dir, r_np):
            # total usable diameter length inside the circle
            diam_len = 2.0 * rad * (1.0 - edge_margin)
            head_len = min(rad * head_frac, diam_len * 0.5)  # keep head inside
            shaft_len = diam_len - head_len

            tail = np.array([cx, cy]) - u * (shaft_len * 0.5 + head_len * 0.5)
            dxdy = u * (shaft_len + head_len)

            arrow = FancyArrow(
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
            ax.add_patch(arrow)

    # ---- rasterise to numpy ----
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.get_renderer().buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


N = 12
pos = torch.rand(N, 2) * 4 - 2
vel = torch.randn(N, 2)
r = torch.full((N,), 0.18)
rgb = torch.randint(0, 256, (N, 3))

frame = render_frame(pos, r, rgb, vel=vel, size=(500, 500), bg=(250, 250, 250))
plt.imshow(frame)
plt.axis("off")
