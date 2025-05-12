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
from tqdm import tqdm


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    print(f"{name:22s}: {time.perf_counter() - t0:6.3f}s")


DEV = "mps" if torch.backends.mps.is_available() else "cpu"
print("Running on:", DEV.upper())

RGB_Int = Tuple[int, int, int]

# ───────────────────────── sprite cache ─────────────────────────────────────
_disc_cache: dict[int, np.ndarray] = {}


def disc_sprite(px_r: int) -> np.ndarray:
    if px_r in _disc_cache:
        return _disc_cache[px_r]
    d = px_r * 2
    im = Image.new("L", (d, d), 0)
    ImageDraw.Draw(im).ellipse((0, 0, d - 1, d - 1), fill=255)
    _disc_cache[px_r] = np.asarray(im)
    return _disc_cache[px_r]


# ───────────────────────── fast renderer ────────────────────────────────────
def render_frame_fast(
    pos: torch.Tensor,
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
        else np.asarray(rgb, np.uint8)
    )

    # discs -------------------------------------------------------------
    for (x, y), rad, col in zip(pos_np, r_np, rgb_np):
        px_r = max(1, int(rad * sx))
        sprite = disc_sprite(px_r)
        d = px_r * 2
        cx = int((x - xmin) * sx)
        cy = int((ymax - y) * sy)
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

    # arrows ------------------------------------------------------------
    if vel is not None:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img, "RGBA")
        vel_np = vel.detach().cpu().numpy()
        for (x, y), v, rad in zip(pos_np, vel_np, r_np):
            n = np.linalg.norm(v)
            0 if n < 1e-6 else None
            u = v / n
            ux, uy = u[0] * sx, -u[1] * sy
            mag = (ux**2 + uy**2) ** 0.5
            ux /= mag
            uy /= mag
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
            w_line = max(1, int(px_r * shaft_width_frac))
            draw.line([tail, head], fill=(255, 255, 255, 230), width=w_line)
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
        neighbor_dist=2.0,  #  ← your new numbers
        separation_dist=0.2,
        cohesion_weight=5.5,
        alignment_weight=5.0,
        separation_weight=20.2,
        max_speed=1.3,
        max_force=2.90,
        world_half_size=8,
    )
    p.update(kw)
    return p


def step(
    s: Dict[str, torch.Tensor], p: Dict[str, float], *, wrap: bool = True
) -> Dict[str, torch.Tensor]:
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
    if wrap:
        half = p["world_half_size"]
        pos = (pos + half) % (2 * half) - half
    return {"pos": pos, "vel": vel, "r": s["r"], "rgb": s["rgb"]}


def rollout(init, p, steps, warmup=0):
    """Run `warmup` steps (discarded) then `steps` steps (returned incl. start)."""
    state = init
    for _ in range(warmup):  # burn‑in
        state = step(state, p)
    traj = [state]
    for _ in range(steps):
        state = step(state, p)
        traj.append(state)
    return traj


def ring_loss(
    final_pos: torch.Tensor, centre: torch.Tensor, radius: float
) -> torch.Tensor:
    """
    Mean‑squared radial error:  (‖x - c‖₂ - R)², averaged over boids.
    """
    r = (final_pos - centre).norm(dim=1)
    return ((r - radius) ** 2).mean()


def warmup_state(
    state: Dict[str, torch.Tensor], p: Dict[str, float], warm_steps: int, device="cpu"
) -> Dict[str, torch.Tensor]:
    """Run `warm_steps` with wrapping ON – returns the *final* state (no grads)."""
    s = {k: v.clone().to(device) for k, v in state.items()}
    for _ in range(warm_steps):
        s = step(s, p, wrap=True)
    # detach everything
    for k in ("pos", "vel"):
        s[k] = s[k].detach()
    return s


def optimise_vel_to_ring(
    state0: Dict[str, torch.Tensor],
    p: Dict[str, float],
    *,
    steps: int = 400,
    R: float = 1.7,
    lr: float = 0.2,
    iters: int = 800,
    device="cpu",
    window: Tuple[float, float, float, float],
    render_every: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    Given a *warmed* state, adjust ONLY the initial velocities so that
    after `steps` frames (no wrapping) the flock sits on a ring of radius `R`.
    Returns the modified initial state (positions unchanged).
    """
    state = {k: v.clone().to(device) for k, v in state0.items()}
    state["vel"].requires_grad_()

    opt = torch.optim.Adam([state["vel"]], lr=lr)
    centre = state["pos"].mean(0).detach()

    for it in (pbar := tqdm(range(iters), desc="optimising velocities")):
        s = state
        for _ in range(steps):
            s = step(s, p, wrap=False)  # smooth gradient path
        loss = ring_loss(s["pos"], centre, R)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # keep within max_speed for physical plausibility
        with torch.no_grad():
            speed = state["vel"].norm(dim=1, keepdim=True).clamp(min=1e-12)
            state["vel"].mul_(p["max_speed"] / speed)

        if it % render_every == 0:
            frame = render_frame_fast(
                s["pos"],
                s["r"],
                s["rgb"],
                s["vel"],
                size=(1024, 1024),
                bg=(0, 0, 0),
                window=window,
            )
            display(Image.fromarray(frame))

        pbar.set_postfix(loss=loss.item())

    # freeze grads before returning
    state["vel"] = state["vel"].detach()
    return state


# ───────────────────────── demo & timings ───────────────────────────────────
# # choose raw random start (pos only, zero vel for clarity) -------------------
N = 400
state_raw = {
    "pos": (torch.rand(N, 2, device=DEV) * 4 - 2),
    "vel": torch.zeros(N, 2, device=DEV),
    "r": torch.full((N,), 0.08),
    "rgb": torch.randint(0, 256, (N, 3), dtype=torch.uint8),
}
p = default_params()
half = p["world_half_size"]
window = (-half, half, -half, half)

# 1. burn‑in to get a plausible configuration -------------------------------
WARM_STEPS = 1000
with timer("warm‑up only"):
    state_warm = warmup_state(state_raw, p, warm_steps=WARM_STEPS, device=DEV)

# 2. optimise just the *velocities* -----------------------------------------
RENDER_STEPS = 100
with timer("optimisation"):
    state_star = optimise_vel_to_ring(
        state_warm,
        p,
        steps=RENDER_STEPS,
        R=1.7,
        lr=0.4,
        iters=100,
        device=DEV,
        window=window,
    )

# 3. run the normal rollout (NO warmup now) ----------------------------------
traj = rollout(state_star, p, steps=RENDER_STEPS, warmup=0)

# 4. render as usual ---------------------------------------------------------
frames = [
    render_frame_fast(
        s["pos"],
        s["r"],
        s["rgb"],
        s["vel"],
        size=(1024, 1024),
        bg=(0, 0, 0),
        window=window,
    )
    for s in traj
]
display(video_from_frames(frames, fps=30))
