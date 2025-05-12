# %% ───────────────────────────── imports & helpers ─────────────────────────
import pathlib, tempfile, time
from contextlib import contextmanager
from typing import Dict, List, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
from IPython.display import Video, display
from tqdm import tqdm

@contextmanager
def timer(name: str):
    t0 = time.perf_counter(); yield
    print(f"{name:22s}: {time.perf_counter() - t0:6.3f}s")

DEV = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("Running on:", DEV.upper())

RGB_Int = Tuple[int, int, int]

# ───────────────────────── sprite cache ─────────────────────────────────────
_disc_cache: dict[int, np.ndarray] = {}
def disc_sprite(px_r: int) -> np.ndarray:
    if px_r in _disc_cache: return _disc_cache[px_r]
    d = px_r * 2
    im = Image.new("L", (d, d), 0)
    ImageDraw.Draw(im).ellipse((0,0,d-1,d-1), fill=255)
    _disc_cache[px_r] = np.asarray(im); return _disc_cache[px_r]

# ───────────────────────── fast renderer ────────────────────────────────────
def render_frame_fast(
    pos: torch.Tensor,                          # [N,2]  (any device)
    r: Sequence[float] | torch.Tensor,          # len N radii
    rgb: Sequence[RGB_Int] | torch.Tensor,      # len N colours
    vel: torch.Tensor | None,                   # [N,2] or None
    size: Tuple[int, int],                      # (W,H) pixels
    bg: RGB_Int,                                # background colour
    *,
    window: Tuple[float, float, float, float],  # (xmin,xmax,ymin,ymax)
    # geometry knobs ----------------------------------------------------------
    edge_margin: float = 0.10,
    head_frac: float = 0.55,
    head_width_frac: float = 0.55,
    shaft_width_frac: float = 0.25,
    # debug overlays ----------------------------------------------------------
    ring: Tuple[float, float, float, RGB_Int, int] | None = None,
    mask: Tuple[torch.Tensor, RGB_Int, float] | None = None,   # (tensor, colour, α)
) -> np.ndarray:                               # returns H×W×3 uint8
    """
    Fast CPU renderer: sprites + Pillow arrows + optional ring & bitmap mask.

    `mask[0]` must be a [1,1,H,W] tensor in 0‑1; it will be bilinear‑resized
    to the output size and alpha‑blended with colour `mask[1]` and opacity
    `mask[2]` (0‑1).
    """
    W, H = size
    xmin, xmax, ymin, ymax = window
    sx = (W - 1) / (xmax - xmin)
    sy = (H - 1) / (ymax - ymin)

    # --- base canvas ---------------------------------------------------------
    canvas = np.empty((H, W, 3), np.uint8)
    canvas[:] = bg

    # --- discs ---------------------------------------------------------------
    pos_np = pos.detach().cpu().numpy()
    r_np   = r.detach().cpu().numpy() if isinstance(r, torch.Tensor) else np.asarray(r)
    rgb_np = (
        torch.as_tensor(rgb).detach().cpu().numpy()
        if isinstance(rgb, torch.Tensor) else np.asarray(rgb, np.uint8)
    )

    for (x, y), rad, col in zip(pos_np, r_np, rgb_np):
        px_r = max(1, int(rad * sx))
        sprite = disc_sprite(px_r)               # α‑mask uint8
        d = px_r * 2
        cx, cy = int((x - xmin) * sx), int((ymax - y) * sy)
        x0, y0 = cx - px_r, cy - px_r
        x1, y1 = x0 + d, y0 + d
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = d - max(0, x1 - W), d - max(0, y1 - H)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        patch = canvas[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1]
        alpha = sprite[sy0:sy1, sx0:sx1, None] / 255.0
        patch[:] = (alpha * col + (1.0 - alpha) * patch).astype(np.uint8)

    # --- velocity arrows -----------------------------------------------------
    if vel is not None:
        img  = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img, "RGBA")
        vel_np = vel.detach().cpu().numpy()

        for (x, y), v, rad in zip(pos_np, vel_np, r_np):
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            u = v / n
            ux, uy = u[0] * sx, -u[1] * sy            # flip y
            mag = (ux * ux + uy * uy) ** 0.5
            ux, uy = ux / mag, uy / mag

            px_r = max(1, int(rad * sx))
            diam = 2 * px_r * (1 - edge_margin)

            tail = ((x - xmin) * sx - ux * diam * 0.5,
                    (ymax - y) * sy - uy * diam * 0.5)
            head = ((x - xmin) * sx + ux * diam * 0.5,
                    (ymax - y) * sy + uy * diam * 0.5)

            w_line = max(1, int(px_r * shaft_width_frac))
            draw.line([tail, head], fill=(255, 255, 255, 230), width=w_line)

            h_len, h_wid = diam * head_frac, px_r * head_width_frac
            p1 = head
            p2 = (head[0] - ux * h_len - uy * h_wid,
                  head[1] - uy * h_len + ux * h_wid)
            p3 = (head[0] - ux * h_len + uy * h_wid,
                  head[1] - uy * h_len - ux * h_wid)
            draw.polygon([p1, p2, p3], fill=(255, 255, 255, 230))

        canvas = np.array(img)            # make writeable copy

    # --- bitmap mask overlay -------------------------------------------------
    if mask is not None:
        mask_tensor, col_mask, alpha_mask = mask        # colour as RGB, alpha 0‑1
        m = mask_tensor.squeeze().detach().cpu().numpy()    # Hm×Wm float
        m_img = Image.fromarray((m * 255).astype(np.uint8), mode="L")
        m_img = m_img.resize(size, resample=Image.BILINEAR)
        m_arr = np.asarray(m_img, dtype=np.float32) / 255.0  # H×W float

        overlay = np.zeros_like(canvas, dtype=np.float32)
        overlay[..., 0] = col_mask[0]
        overlay[..., 1] = col_mask[1]
        overlay[..., 2] = col_mask[2]

        a = (alpha_mask * m_arr)[..., None]              # H×W×1
        canvas = (canvas * (1.0 - a) + overlay * a).astype(np.uint8)

    # --- ring overlay (draw on top) -----------------------------------------
    if ring is not None:
        cx_w, cy_w, R_w, col_ring, wpx = ring
        cx, cy = (cx_w - xmin) * sx, (ymax - cy_w) * sy
        Rp = R_w * sx
        img = Image.fromarray(canvas.copy())
        ImageDraw.Draw(img).ellipse((cx - Rp, cy - Rp, cx + Rp, cy + Rp),
                                    outline=col_ring, width=wpx)
        canvas = np.asarray(img)

    return canvas


def video_from_frames(frames:List[np.ndarray],fps=30,path=None)->Video:
    path=pathlib.Path(path or tempfile.mktemp(suffix=".mp4"))
    with imageio.get_writer(path,fps=fps,codec="libx264",quality=8) as w:
        for f in frames:w.append_data(f)
    return Video(str(path),embed=True)

# ───────────────────────── boids core ───────────────────────────────────────
def default_params(**kw)->Dict[str,float]:
    p=dict(dt=0.03, neighbor_dist=2.0, separation_dist=0.2,
           cohesion_weight=5.5, alignment_weight=5.0, separation_weight=20.2,
           max_speed=1.3, max_force=2.9, world_half_size=8)
    p.update(kw); return p

def step(s:Dict[str,torch.Tensor],p:Dict[str,float],*,wrap=True)->Dict[str,torch.Tensor]:
    pos,vel=s["pos"],s["vel"]; dt,vmax,fmax=p["dt"],p["max_speed"],p["max_force"]
    nd,sd=p["neighbor_dist"],p["separation_dist"]
    d=pos[:,None]-pos[None]; dist=d.norm(dim=-1)+1e-12
    neigh=(0<dist)&(dist<nd); sep_m=(0<dist)&(dist<sd); cnt=neigh.sum(1,keepdim=True).clamp(min=1)
    centre=(neigh.float().unsqueeze(-1)*pos[None]).sum(1)/cnt
    coh=(centre-pos)*p["cohesion_weight"]
    mean_v=(neigh.float().unsqueeze(-1)*vel[None]).sum(1)/cnt
    ali=(mean_v-vel)*p["alignment_weight"]
    sep=(-d/dist.unsqueeze(-1)).masked_fill(~sep_m.unsqueeze(-1),0).sum(1)*p["separation_weight"]
    acc=coh+ali+sep; acc=acc/acc.norm(dim=1,keepdim=True).clamp(min=1e-12)*fmax
    vel=vel+acc*dt; vel=vel/vel.norm(dim=1,keepdim=True).clamp(min=1e-12)*vmax
    pos=pos+vel*dt
    if wrap:
        half=p["world_half_size"]; pos=(pos+half)%(2*half)-half
    return {"pos":pos,"vel":vel,"r":s["r"],"rgb":s["rgb"]}

def rollout(init,p,steps,warmup=0):
    state=init
    for _ in range(warmup): state=step(state,p)
    traj=[state]
    for _ in range(steps):
        state=step(state,p); traj.append(state)
    return traj

# ───────────────────────── optimisation utils ───────────────────────────────


def warmup_state(state,p,warm_steps,device="cpu"):
    s={k:v.clone().to(device) for k,v in state.items()}
    for _ in range(warm_steps): s=step(s,p)
    for k in("pos","vel"): s[k]=s[k].detach(); return s

def render_every_n(traj,n,*,window,size,bg,ring=None):
    last=len(traj)-1; frames=[]
    for i,s in enumerate(traj):
        if i%n==0 or i==last:
            frames.append(
                render_frame_fast(s["pos"],s["r"],s["rgb"],s["vel"],
                                  size=size,bg=bg,window=window,ring=ring)
            )
    return frames


from PIL import ImageFont, ImageOps, ImageFilter

def make_text_mask(text: str,
                   canvas_px: int = 256,
                   fill_frac: float = 0.4,        # 0‑1 of canvas height
                   font_path: str | None = None) -> torch.Tensor:
    """
    Returns [1,1,H,W] float mask.
    `fill_frac` ≈ target letter height / canvas_px.
    """
    font_size = int(canvas_px * fill_frac)
    font      = ImageFont.truetype(font_path or "DejaVuSans-Bold.ttf", font_size)

    img = Image.new("L", (canvas_px, canvas_px), 0)
    draw = ImageDraw.Draw(img)

    try:
        w, h = draw.textsize(text, font=font)
    except AttributeError:                          # Pillow ≥ 10
        box = draw.textbbox((0,0), text, font=font)
        w, h = box[2]-box[0], box[3]-box[1]

    draw.text(((canvas_px-w)//2, (canvas_px-h)//2), text, 255, font=font)
    img = img.filter(ImageFilter.GaussianBlur(1))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.tensor(arr)[None,None]              # [1,1,H,W]

def text_loss(final_pos, mask, world_half_size):
    """
    final_pos : [N,2]  (world coords)
    mask      : [1,1,H,W] 0‑1
    """
    H,W = mask.shape[-2:]
    # world → grid_sample coords [-1,1]
    grid = final_pos.clone()
    grid[:,0] = (grid[:,0] / world_half_size).clamp(-1,1)
    grid[:,1] = (-grid[:,1] / world_half_size).clamp(-1,1)

    grid = grid.view(1,-1,1,2)                # [1,N,1,2]
    samples = torch.nn.functional.grid_sample(mask.to(grid), grid,
                                              align_corners=True).view(-1)
    return (1.0 - samples).mean()               # want samples ≈ 1

def optimise_vel_to_text(state0, p, *, text="HELLO", steps=400,
                          lr=0.3, iters=600, preview=100, device="cpu", canvas_px=256, fill_frac=0.4):
    mask = make_text_mask(text, canvas_px=canvas_px, fill_frac=fill_frac).to(device)
    mask_overlay = (mask, (0, 255, 0), 0.25)
    wh = p["world_half_size"]
    state = {k:v.clone().to(device) for k,v in state0.items()}
    state["vel"].requires_grad_()
    opt = torch.optim.Adam([state["vel"]], lr=lr)

    for it in (pbar:=tqdm(range(iters), desc="optimising text")):
        s = state
        for _ in range(steps):
            s = step(s, p, wrap=False)
        loss = text_loss(s["pos"], mask, wh)

        opt.zero_grad(); loss.backward(); opt.step()

        # clamp speed
        with torch.no_grad():
            speed = state["vel"].norm(dim=1, keepdim=True).clamp(min=1e-12)
            state["vel"].mul_(p["max_speed"]/speed)

        pbar.set_postfix(loss=loss.item())
        if it % preview == 0:
            frame = render_frame_fast(
                s["pos"],
                s["r"],
                s["rgb"],
                s["vel"],
                size=(600,600),
                bg=(0,0,0),
                mask=mask_overlay,
                window=(-wh,wh,-wh,wh)
            )
            display(Image.fromarray(frame))

    state["vel"] = state["vel"].detach()
    return state

@torch.no_grad()
def random_search_velocities(
    state_base: Dict[str, torch.Tensor],
    p: Dict[str, float],
    mask: torch.Tensor,
    *,
    tries: int = 4000,
    steps: int = 300,
    batch: int = 64,          # evaluate this many random fields at once
    device: str | torch.device = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Quickly evaluates `tries` random velocity initialisations (no autograd).
    Returns a *clone* of `state_base` with the best‑loss velocity field.
    """
    world_half = p["world_half_size"]
    N = state_base["pos"].shape[0]

    best_loss = float("inf")
    best_vel  = None

    # broadcast constant position once
    pos_const = state_base["pos"].to(device)  # [N,2]

    # small utility to run vectorised steps on shape [B,N,2]
    def batch_step(pos, vel):
        # pos, vel: [B,N,2]
        dt, vmax, fmax = p["dt"], p["max_speed"], p["max_force"]
        nd, sd = p["neighbor_dist"], p["separation_dist"]

        d = pos[:, :, None] - pos[:, None]           # [B,N,N,2]
        dist = d.norm(dim=-1) + 1e-12

        neigh = (0 < dist) & (dist < nd)
        sep_m = (0 < dist) & (dist < sd)
        cnt = neigh.sum(-1, keepdim=True).clamp(min=1)

        centre = (neigh[..., None] * pos[:, None]).sum(2) / cnt
        coh = (centre - pos) * p["cohesion_weight"]

        mean_v = (neigh[..., None] * vel[:, None]).sum(2) / cnt
        ali = (mean_v - vel) * p["alignment_weight"]

        sep = (-d / dist[..., None]).masked_fill(
            ~sep_m[..., None], 0).sum(2) * p["separation_weight"]

        acc = coh + ali + sep
        acc = acc / acc.norm(dim=-1, keepdim=True).clamp(min=1e-12) * fmax

        vel = vel + acc * dt
        vel = vel / vel.norm(dim=-1, keepdim=True).clamp(min=1e-12) * vmax
        pos = pos + vel * dt                          # wrapping OFF

        return pos, vel

    mask = mask.to(device)

    for start in range(0, tries, batch):
        B = min(batch, tries - start)
        # generate B random velocity fields, norm == max_speed
        vel = torch.randn(B, N, 2, device=device)
        speed = vel.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        vel = vel / speed * p["max_speed"]

        pos = pos_const.expand(B, -1, -1).clone()     # [B,N,2]

        for _ in range(steps):
            pos, vel = batch_step(pos, vel)

        # evaluate loss per batch element
        loss_b = text_loss(pos.reshape(-1, 2), mask,
                           world_half).view(B, -1).mean(1)  # [B]

        min_val, idx = loss_b.min(0)
        if min_val < best_loss:
            best_loss = min_val.item()
            best_vel  = vel[idx].detach().cpu()

    print(f"best random loss: {best_loss:.4f}")
    best_state = {k: v.clone() for k, v in state_base.items()}
    best_state["vel"] = best_vel.to(state_base["vel"])
    return best_state

# ───────────────────────── demo & timings  ──────────────────────────────────
torch.manual_seed(0)
# 1) create mask first (same call your optimiser uses)
mask = make_text_mask("Hi", canvas_px=1024, fill_frac=0.4).to(DEV)

# random start
N = 400

state0 = {
    "pos": (torch.rand(N,2,device=DEV)*4-2),
    "vel": torch.zeros(N,2,device=DEV),
    "r"  : torch.full((N,), 0.08),
    "rgb": torch.randint(0,256,(N,3),dtype=torch.uint8),
}

p = default_params()
state_warm = warmup_state(state0, p, warm_steps=500, device=DEV)

# 2) quick random search (~ a few seconds on GPU)
state_seed = random_search_velocities(
    state_warm,          # warmed positions, zero vel
    p,
    mask,
    tries=4000,          # explore 4k random velocity fields
    steps=300,           # simulate same horizon as optimiser
    batch=128,
    device=DEV,
)



with timer("optimise text"):
    state_txt = optimise_vel_to_text(state_warm, p,
                                     text="Hi", steps=300,
                                     lr=0.4, iters=800, preview=100,
                                     device=DEV, canvas_px=1024, fill_frac=0.4)

traj = rollout(state_txt, p, steps=300, warmup=0)
frames = render_every_n(traj, 5, window=(-p["world_half_size"])*np.array([1,-1,1,-1]),
                        size=(600,600), bg=(0,0,0))
display(video_from_frames(frames, fps=30))
