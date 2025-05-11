# %%  ───────────────────────────── imports & helpers ─────────────────────────
import pathlib
import tempfile
import time
from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from IPython.display import Video, display
from PIL import Image, ImageDraw


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    print(f"{name:18s}: {time.perf_counter() - t0:6.3f}s")


DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print("Running on:", DEV.upper())

RGB_Int = Tuple[int, int, int]

# ───────────────────────── sprite cache ─────────────────────────────────────
_disc_cache: dict[int, np.ndarray] = {}


def disc_sprite(px_r: int) -> np.ndarray:  # uint8 mask
    if px_r in _disc_cache:
        return _disc_cache[px_r]
    d = px_r * 2
    im = Image.new("L", (d, d), 0)
    ImageDraw.Draw(im).ellipse((0, 0, d - 1, d - 1), fill=255)
    _disc_cache[px_r] = np.asarray(im)
    return _disc_cache[px_r]


# ───────────────────────── fast renderer ────────────────────────────────────
def render_frame_fast(
    pos: torch.Tensor,  # [N,2] (device→CPU inside)
    r: Sequence[float] | torch.Tensor,
    rgb: Sequence[RGB_Int] | torch.Tensor,
    vel: torch.Tensor | None,
    size: Tuple[int, int],
    bg: RGB_Int,
    *,
    window: Tuple[float, float, float, float],
    edge_margin=0.10,
    head_frac=0.55,
    head_width_frac=0.55,
    shaft_width_frac=0.25,
) -> np.ndarray:
    W, H = size
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = bg
    xmin, xmax, ymin, ymax = window
    sx = (W - 1) / (xmax - xmin)
    sy = (H - 1) / (ymax - ymin)

    pos_np = pos.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
    rgb_np = (
        torch.as_tensor(rgb).detach().cpu().numpy()
        if isinstance(rgb, torch.Tensor)
        else np.asarray(rgb, dtype=np.uint8)
    )

    # --- discs --------------------------------------------------------------
    for (x, y), rad, col in zip(pos_np, r_np, rgb_np):
        px_r = max(1, int(rad * sx))
        sprite = disc_sprite(px_r)
        d = px_r * 2
        cx = int((x - xmin) * sx)
        cy = int((ymax - y) * sy)  # y‑flip
        x0, y0 = cx - px_r, cy - px_r
        x1, y1 = x0 + d, y0 + d
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = d - max(0, x1 - W)
        sy1 = d - max(0, y1 - H)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        patch = canvas[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1]
        alpha = sprite[sy0:sy1, sx0:sx1, None] / 255.0
        patch[:] = (alpha * col + (1 - alpha) * patch).astype(np.uint8)

    # --- velocity arrows ----------------------------------------------------
    if vel is not None:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img, "RGBA")
        vel_np = vel.detach().cpu().numpy()
        for (x, y), v, rad in zip(pos_np, vel_np, r_np):
            norm = np.linalg.norm(v)
            if norm < 1e-6:
                continue  # skip stagnating
            u = v / norm
            # pixel‑space unit vector (note y flip)
            ux, uy = u[0] * sx, -u[1] * sy
            mag = np.sqrt(ux * ux + uy * uy)
            ux, uy = ux / mag, uy / mag
            px_r = max(1, int(rad * sx))
            diam = 2 * px_r * (1 - edge_margin)
            tail = (
                (x - xmin) * sx - ux * diam * 0.5,
                (ymax - y) * sy - uy * diam * 0.5,
            )
            head = (
                (x - xmin) * sx + ux * diam * 0.5,
                (ymax - y) * sy + uy * diam * 0.5,
            )
            # shaft
            w_line = max(1, int(px_r * shaft_width_frac))
            draw.line([tail, head], fill=(255, 255, 255, 230), width=w_line)
            # triangular head
            h_len = diam * head_frac
            h_wid = px_r * head_width_frac
            p1 = head
            p2 = (head[0] - ux * h_len + -uy * h_wid, head[1] - uy * h_len + ux * h_wid)
            p3 = (head[0] - ux * h_len - -uy * h_wid, head[1] - uy * h_len - ux * h_wid)
            draw.polygon([p1, p2, p3], fill=(255, 255, 255, 230))
        canvas = np.asarray(img)
    return canvas


def video_from_frames(frames: List[np.ndarray], fps=30, path=None) -> Video:
    path = pathlib.Path(path or tempfile.mktemp(suffix=".mp4"))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    return Video(str(path), embed=True)


# ───────────────────────── boids core ───────────────────────────────────────
def default_params(**kw) -> Dict[str, float]:
    p = dict(
        dt=0.03,
        neighbor_dist=4.2,
        separation_dist=0.15,
        cohesion_weight=3.5,
        alignment_weight=10.0,
        separation_weight=1.2,
        max_speed=1.0,
        max_force=0.50,
        world_half_size=2.5,
    )
    p.update(kw)
    return p


def step(s: Dict[str, torch.Tensor], p: Dict[str, float]) -> Dict[str, torch.Tensor]:
    pos, vel = s["pos"], s["vel"]
    dt, vmax, fmax = p["dt"], p["max_speed"], p["max_force"]
    nd, sd = p["neighbor_dist"], p["separation_dist"]
    d = pos[:, None] - pos[None]
    dist = d.norm(dim=-1) + 1e-12
    neigh = (0 < dist) & (dist < nd)
    sep_m = (0 < dist) & (dist < sd)
    cnt = neigh.sum(1, keepdim=True).clamp(min=1)
    centre = (neigh.float().unsqueeze(-1) * pos[None]).sum(1) / cnt
    coh = (centre - pos) * p["cohesion_weight"]
    mean_v = (neigh.float().unsqueeze(-1) * vel[None]).sum(1) / cnt
    ali = (mean_v - vel) * p["alignment_weight"]
    sep = (-d / dist.unsqueeze(-1)).masked_fill(~sep_m.unsqueeze(-1), 0).sum(1) * p[
        "separation_weight"
    ]
    acc = coh + ali + sep
    acc = acc / acc.norm(dim=1, keepdim=True).clamp(min=1e-12) * fmax
    vel = vel + acc * dt
    vel = vel / vel.norm(dim=1, keepdim=True).clamp(min=1e-12) * vmax
    pos = pos + vel * dt
    half = p["world_half_size"]
    pos = (pos + half) % (2 * half) - half
    return {"pos": pos, "vel": vel, "r": s["r"], "rgb": s["rgb"]}


def rollout(init, p, steps):
    traj = [init]
    for _ in range(steps):
        traj.append(step(traj[-1], p))
    return traj


# ───────────────────────── demo & timings ───────────────────────────────────
torch.manual_seed(0)
N, STEPS = 50, 400
state0 = {
    "pos": (torch.rand(N, 2, device=DEV) * 5 - 2.5),
    "vel": torch.randn(N, 2, device=DEV),
    "r": torch.full((N,), 0.12),  # CPU tensors for sprites
    "rgb": torch.randint(0, 256, (N, 3), dtype=torch.uint8),
}
p = default_params()
_ = step(state0, p)  # JIT warm‑up

with timer("simulation"):
    traj = rollout(state0, p, STEPS)

half = p["world_half_size"]
window = (-half, half, -half, half)

with timer("rendering (Pillow)"):
    frames = [
        render_frame_fast(
            s["pos"],
            s["r"],
            s["rgb"],
            s["vel"],
            size=(512, 512),
            bg=(250, 250, 250),
            window=window,
        )
        for s in traj
    ]

with timer("encoding"):
    vid = video_from_frames(frames, fps=30)

display(vid)
