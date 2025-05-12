# %% ────────────────────────── imports & helpers ────────────────────────────
import time, tempfile, pathlib, warnings
from contextlib import contextmanager
from typing import Dict, Tuple, List, Sequence

import numpy as np
import torch, imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from IPython.display import Video, display
from tqdm.auto import tqdm

# external ‑‑ install once:  pip install scipy
from scipy.ndimage import distance_transform_edt

DEV = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Running on:", DEV.upper())

@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    print(f"{name:20s}: {time.perf_counter() - t0:6.3f}s")

# type aliases
RGB_Int = Tuple[int, int, int]

_disc_cache: dict[int, np.ndarray] = {}

def disc_sprite(px_r: int) -> np.ndarray:
    if px_r not in _disc_cache:
        d = px_r * 2
        im = Image.new("L", (d, d), 0)
        ImageDraw.Draw(im).ellipse((0, 0, d - 1, d - 1), fill=255)
        _disc_cache[px_r] = np.asarray(im)
    return _disc_cache[px_r]

def render_frame_fast(
    pos: torch.Tensor,  # [N,2]
    r: torch.Tensor,    # [N]
    rgb: torch.Tensor,  # [N,3] uint8
    vel: torch.Tensor | None,
    size: Tuple[int, int],
    bg: RGB_Int,
    *,
    window: Tuple[float, float, float, float],
    edge_margin: float = 0.10,
    head_frac: float = 0.55,
    head_width_frac: float = 0.55,
    shaft_width_frac: float = 0.25,
    mask: Tuple[torch.Tensor, RGB_Int, float] | None = None,
    ring: Tuple[int, int, int, RGB_Int, int] | None = None,
) -> np.ndarray:
    W, H = size
    xmin, xmax, ymin, ymax = window
    sx, sy = (W - 1) / (xmax - xmin), (H - 1) / (ymax - ymin)

    canvas = np.full((H, W, 3), bg, np.uint8)

    pos_np = pos.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    rgb_np = rgb.detach().cpu().numpy()

    # --- discs
    for (x, y), rad, col in zip(pos_np, r_np, rgb_np):
        px_r = max(1, int(rad * sx))
        sprite = disc_sprite(px_r)
        d = px_r * 2
        cx, cy = int((x - xmin) * sx), int((ymax - y) * sy)
        x0, y0 = cx - px_r, cy - px_r
        x1, y1 = x0 + d, y0 + d
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = d - max(0, x1 - W), d - max(0, y1 - H)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        patch = canvas[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1]
        a = sprite[sy0:sy1, sx0:sx1, None] / 255.0
        patch[:] = (a * col + (1 - a) * patch).astype(np.uint8)

    # --- arrows
    if vel is not None:
        img = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img, "RGBA")
        vel_np = vel.detach().cpu().numpy()
        for (x, y), v, rad in zip(pos_np, vel_np, r_np):
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            u = v / n
            ux, uy = u[0] * sx, -u[1] * sy
            m = np.hypot(ux, uy)
            ux, uy = ux / m, uy / m
            px_r = max(1, int(rad * sx))
            diam = 2 * px_r * (1 - edge_margin)
            tail = ((x - xmin) * sx - ux * diam * 0.5, (ymax - y) * sy - uy * diam * 0.5)
            head = ((x - xmin) * sx + ux * diam * 0.5, (ymax - y) * sy + uy * diam * 0.5)
            draw.line(
                [tail, head],
                fill=(255, 255, 255, 230),
                width=max(1, int(px_r * shaft_width_frac)),
            )
            h_len, h_wid = diam * head_frac, px_r * head_width_frac
            p1 = head
            p2 = (head[0] - ux * h_len - uy * h_wid, head[1] - uy * h_len + ux * h_wid)
            p3 = (head[0] - ux * h_len + uy * h_wid, head[1] - uy * h_len - ux * h_wid)
            draw.polygon([p1, p2, p3], fill=(255, 255, 255, 230))
        canvas = np.asarray(img)

    # --- mask overlay
    if mask is not None:
        m_t, col, a = mask
        m = m_t.squeeze().cpu().numpy()
        m_img = Image.fromarray((m * 255).astype(np.uint8)).resize(size, Image.BILINEAR)
        m_arr = np.asarray(m_img, np.float32) / 255.0
        overlay = np.zeros_like(canvas, dtype=np.float32)
        overlay[..., 0], overlay[..., 1], overlay[..., 2] = col
        alpha = (a * m_arr)[..., None]
        canvas = (canvas * (1 - alpha) + overlay * alpha).astype(np.uint8)

    # --- optional ring
    if ring is not None:
        cx, cy, R, col, wpx = ring
        img = Image.fromarray(canvas)
        ImageDraw.Draw(img).ellipse((cx - R, cy - R, cx + R, cy + R), outline=col, width=wpx)
        canvas = np.asarray(img)

    return canvas

def video_from_frames(frames: Sequence[np.ndarray], fps: int = 30) -> Video:
    path = pathlib.Path(tempfile.mktemp(suffix=".mp4"))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as w:
        for f in frames:
            w.append_data(f)
    return Video(str(path), embed=True)

def default_params(**kw) -> Dict[str, float]:
    p = dict(
        dt=0.03,
        neighbor_dist=2.0,
        separation_dist=0.25,
        cohesion_weight=1.0,
        alignment_weight=1.0,
        separation_weight=1.5,
        max_speed=1.2,
        max_force=1.5,
        world_half_size=8.0,
    )
    p.update(kw)
    return p

def step(state: Dict[str, torch.Tensor], p: Dict[str, float], *, wrap: bool = True
         ) -> Dict[str, torch.Tensor]:
    pos, vel = state["pos"], state["vel"]                 # [N,2]
    dt, vmax, fmax = p["dt"], p["max_speed"], p["max_force"]
    nd, sd = p["neighbor_dist"], p["separation_dist"]

    # pairwise deltas & distances
    d = pos[:, None] - pos[None]                          # [N,N,2]
    dist = d.norm(dim=-1) + 1e-12                         # [N,N]
    neigh = (0 < dist) & (dist < nd)
    sep_m = (0 < dist) & (dist < sd)
    cnt = neigh.sum(-1, keepdim=True).clamp(min=1)        # [N,1]

    centre   = (neigh[..., None] * pos[None]).sum(1) / cnt
    coh      = (centre - pos) * p["cohesion_weight"]

    mean_vel = (neigh[..., None] * vel[None]).sum(1) / cnt
    ali      = (mean_vel - vel) * p["alignment_weight"]

    sep_vec  = (-d / dist[..., None]).masked_fill(~sep_m[..., None], 0.0).sum(1)
    sep      = sep_vec * p["separation_weight"]

    acc = coh + ali + sep
    acc_len = acc.norm(dim=1, keepdim=True).clamp(min=1e-12)
    acc = acc / acc_len * acc_len.clamp(max=fmax)

    vel = vel + acc * dt
    spd = vel.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vel = vel / spd * spd.clamp(max=vmax)

    pos = pos + vel * dt
    if wrap:
        h = p["world_half_size"]
        pos = (pos + h) % (2 * h) - h

    return {"pos": pos, "vel": vel, "r": state["r"], "rgb": state["rgb"]}

def rollout(init: Dict[str, torch.Tensor], p: Dict[str, float], steps: int
            ) -> List[Dict[str, torch.Tensor]]:
    traj = [init]
    for _ in range(steps):
        traj.append(step(traj[-1], p))
    return traj

def make_text_mask(
    text: str,
    canvas_px: int = 256,
    fill_frac: float = 0.4,
    font_path: str | None = None,
) -> torch.Tensor:
    font = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", int(canvas_px * fill_frac))
    img = Image.new("L", (canvas_px, canvas_px), 0)
    draw = ImageDraw.Draw(img)
    try:
        w, h = draw.textsize(text, font=font)
    except AttributeError:                                # Pillow ≥10
        box = draw.textbbox((0, 0), text, font=font)
        w, h = box[2] - box[0], box[3] - box[1]
    draw.text(((canvas_px - w) // 2, (canvas_px - h) // 2), text, 255, font=font)
    img = img.filter(ImageFilter.GaussianBlur(1))
    return torch.tensor(np.array(img, np.float32) / 255.0)[None, None]

def text_sdf(
    text: str,
    canvas_px: int = 512,
    fill_frac: float = 0.35,
    font_path: str | None = None,
) -> torch.Tensor:
    """
    1×1×H×W signed‑distance field, normalised to roughly [‑1,1]
    (+ve outside, ‑ve inside glyph).
    """
    mask = make_text_mask(text, canvas_px, fill_frac, font_path).squeeze()  # (H,W)
    inside = distance_transform_edt(mask.numpy())
    outside = distance_transform_edt((1 - mask).numpy())
    sdf = torch.from_numpy(outside - inside).float()
    sdf /= sdf.abs().max().clamp(1)
    return sdf[None, None]

def sdf_loss(
    pos: torch.Tensor,
    sdf: torch.Tensor,
    half: float,
    margin: float = 0.02,
) -> torch.Tensor:
    g = pos.clone()
    g[:, 0] = ( g[:, 0] / half).clamp(-1, 1)
    g[:, 1] = (-g[:, 1] / half).clamp(-1, 1)
    d = torch.nn.functional.grid_sample(sdf, g.view(1, -1, 1, 2), align_corners=True).view(-1)
    return torch.relu(d + margin).mean()                  # 0 inside, ↑ outside

def spread_loss(pos: torch.Tensor, half: float) -> torch.Tensor:
    d = pos[:, None] - pos[None]
    dist = d.norm(dim=-1) + 1e-8
    m = (dist > 0).float()
    ideal = (2 * half / pos.size(0)) ** 0.5              # heuristic
    return (torch.relu(ideal - dist) * m).sum() / m.sum().clamp(1)

def optimise_state_to_text(
    N: int,
    p: Dict[str, float],
    *,
    text: str = "HELLO",
    steps: int = 300,
    iters: int = 1000,
    lr: float = 0.3,
    preview: int = 100,
    canvas_px: int = 512,
    fill_frac: float = 0.35,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    half = p["world_half_size"]

    # --- target field & pretty preview mask
    sdf = text_sdf(text, canvas_px, fill_frac).to(device)
    mask_overlay = (torch.relu(-sdf).squeeze(), (0, 255, 0), 0.35)

    # --- learnable parameters
    pos = torch.nn.Parameter(torch.rand(N, 2, device=device) * 2 * half - half)
    vel = torch.nn.Parameter(torch.randn(N, 2, device=device))
    vel.data = vel.data / vel.data.norm(dim=1, keepdim=True).clamp(min=1e-12) * p["max_speed"]

    r   = torch.full((N,), 0.08, device=device)
    rgb = torch.randint(0, 256, (N, 3), dtype=torch.uint8, device="cpu")

    opt = torch.optim.Adam([pos, vel], lr=lr)

    for it in (bar := tqdm(range(iters), desc="optimising")):
        state = {"pos": pos, "vel": vel, "r": r, "rgb": rgb}
        s = state
        for _ in range(steps):
            s = step(s, p, wrap=False)

        loss_main   = sdf_loss(s["pos"], sdf, half, margin=0.02)
        loss_spread = 0.01 * spread_loss(s["pos"], half)
        loss = loss_main + loss_spread

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # clamp
        with torch.no_grad():
            pos.clamp_(-half, half)
            vel.mul_(p["max_speed"] / vel.norm(dim=1, keepdim=True).clamp(min=1e-12))

        bar.set_postfix(loss=loss.item())
        if it % preview == 0 or it == iters - 1:
            frame = render_frame_fast(
                s["pos"], r, rgb, s["vel"],
                (600, 600), (0, 0, 0),
                window=(-half, half, -half, half),
                mask=mask_overlay,
            )
            display(Image.fromarray(frame))

    return {"pos": pos.detach(), "vel": vel.detach(), "r": r, "rgb": rgb}

if __name__ == "__main__":
    torch.manual_seed(0)

    N = 400
    STEPS = 1
    ITERS = 800
    params = default_params()

    with timer("optimise to text"):
        state = optimise_state_to_text(
            N,
            params,
            text="Hi!",
            steps=STEPS,
            iters=ITERS,
            lr=0.3,
            preview=200,
            canvas_px=768,
            fill_frac=0.32,
            device=DEV,
        )

    traj = rollout(state, params, steps=STEPS)
    frames = [
        render_frame_fast(
            s["pos"], s["r"], s["rgb"], s["vel"],
            (600, 600), (0, 0, 0),
            window=(
                -params["world_half_size"],
                params["world_half_size"],
                -params["world_half_size"],
                params["world_half_size"],
            ),
        )
        for i, s in enumerate(traj)
        if (i % 5 == 0 or i == len(traj) - 1)
    ]

    display(video_from_frames(frames, fps=30))
