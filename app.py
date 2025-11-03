import numpy as np
import random
from dataclasses import dataclass
from typing import List, Tuple

import panel as pn
from bokeh.plotting import figure
from bokeh.events import Tap

# ----------------------------
# Palettes (first = your 5)
# ----------------------------
PALETTES = [
    ("Five (green/blue/pink/yellow/white)", [
        (0.60, 1.00, 0.60, 1.0),
        (0.60, 0.80, 1.00, 1.0),
        (1.00, 0.60, 0.80, 1.0),
        (1.00, 1.00, 0.60, 1.0),
        (1.00, 1.00, 1.00, 1.0),
    ]),
    ("Warm Gouache", [
        (0.99, 0.73, 0.60, 1.0),
        (0.96, 0.56, 0.67, 1.0),
        (0.99, 0.87, 0.60, 1.0),
        (0.85, 0.40, 0.36, 1.0),
        (1.00, 1.00, 1.00, 1.0),
    ]),
    ("Cool Mist", [
        (0.70, 0.88, 1.00, 1.0),
        (0.63, 0.98, 0.85, 1.0),
        (0.78, 0.78, 1.00, 1.0),
        (0.56, 0.94, 0.94, 1.0),
        (1.00, 1.00, 1.00, 1.0),
    ]),
]

def get_palette(idx):
    idx = int(np.clip(idx, 0, len(PALETTES)-1))
    return PALETTES[idx]

# ----------------------------
# Geometry helpers
# ----------------------------
def chaikin_smooth(x, y, rounds=2, closed=True):
    pts = np.column_stack([x, y])
    if closed:
        pts = np.vstack([pts, pts[0]])
    for _ in range(rounds):
        Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        R = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.empty((Q.shape[0]+R.shape[0], 2))
        pts[0::2], pts[1::2] = Q, R
        if closed:
            pts = np.vstack([pts, pts[0]])
    if closed:
        pts = pts[:-1]
    return pts[:,0], pts[:,1]

# Shapes
def make_blob(n_points=220, radius=1.0, wobble=0.35, irregularity=0.15):
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    k = random.choice([2,3,4,5,7])
    ripple = wobble * np.sin(k*angles + random.random()*2*np.pi)
    noise = irregularity * np.random.randn(n_points)
    r = np.clip(radius * (1 + ripple + noise), 0.05, None)
    x, y = r*np.cos(angles), r*np.sin(angles)
    return chaikin_smooth(x, y, rounds=2, closed=True)

def make_heart(n_points=360, scale=1.0, offset=(0,0)):
    t = np.linspace(0, 2*np.pi, n_points)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    x, y = scale*x + offset[0], scale*y + offset[1]
    return chaikin_smooth(x, y, rounds=1, closed=True)

# ----------------------------
# Drawing helpers (Bokeh)
# ----------------------------
def rgba_to_hex_alpha(rgba: Tuple[float, float, float, float]) -> Tuple[str, float]:
    r, g, b, a = rgba
    r255 = int(np.clip(round(r*255), 0, 255))
    g255 = int(np.clip(round(g*255), 0, 255))
    b255 = int(np.clip(round(b*255), 0, 255))
    return f"#{r255:02x}{g255:02x}{b255:02x}", float(a)

def golden_angle_positions(n, radius_min=0.6, radius_max=2.6):
    phi = np.pi*(3 - np.sqrt(5))
    centers, radii = [], []
    for i in range(n):
        r = np.interp(i, [0, max(1, n-1)], [radius_min, radius_max])
        theta = i * phi
        cx, cy = 0.35*r*np.cos(theta), 0.35*r*np.sin(theta)
        centers.append((cx, cy)); radii.append(r)
    return centers, radii

def draw_scene(fig, blobs, hearts, add_grain=True):
    # Remove previous glyph renderers (keep axes/grids)
    fig.renderers = [r for r in fig.renderers if getattr(r, "level", "") in ("underlay", "annotation")]

    # Background grain
    if add_grain:
        H, W = 200, 200
        noise = np.random.rand(H, W)
        fig.image(image=[noise], x=-4, y=-4, dw=8, dh=8, palette="Greys256", alpha=0.10)

    # Blobs
    for b in blobs:
        x, y = make_blob(n_points=b["points"], radius=b["radius"],
                         wobble=b["wobble"], irregularity=0.12)
        x, y = x + b["cx"], y + b["cy"]
        color_hex, _ = rgba_to_hex_alpha(b["color"])
        fig.patch(x, y, fill_color=color_hex, fill_alpha=0.55, line_color=None)

    # Hearts
    for h in hearts:
        hx, hy = make_heart(n_points=h["points"], scale=h["scale"], offset=(h["cx"], h["cy"]))
        color_hex, _ = rgba_to_hex_alpha(h["color"])
        fig.patch(hx, hy, fill_color=color_hex, fill_alpha=0.75, line_color=None)

# ----------------------------
# Data factories
# ----------------------------
def make_default_blobs(n, wobble, palette, rng):
    blobs = []
    centers, radii = golden_angle_positions(n)
    for i, ((cx,cy), r) in enumerate(zip(centers, radii)):
        blobs.append({
            "cx": float(cx), "cy": float(cy),
            "radius": float(r),
            "wobble": float(wobble * rng.uniform(0.85, 1.2)),
            "color": palette[i % len(palette)],
            "points": int(rng.integers(200, 260))
        })
    return blobs

def make_default_hearts(n, rng):
    heart_palette = [
        (1.0, 0.4, 0.7, 1.0), (1.0, 0.2, 0.5, 1.0),
        (0.9, 0.3, 0.9, 1.0), (1.0, 0.85, 0.3, 1.0),
        (0.6, 0.9, 1.0, 1.0), (0.6, 1.0, 0.6, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]
    hearts = []
    for _ in range(n):
        scale = rng.uniform(0.05, 0.12)
        cx, cy = rng.uniform(-3.2, 3.2), rng.uniform(-3.2, 3.2)
        hearts.append({
            "cx": float(cx), "cy": float(cy),
            "scale": float(scale),
            "color": tuple(rng.choice(heart_palette)),
            "points": int(rng.integers(260, 340))
        })
    return hearts

# ----------------------------
# Web App (Panel + Bokeh)
# ----------------------------
@dataclass
class ShapeStore:
    base_blobs: List[dict]
    base_hearts: List[dict]
    added: List[dict]

class PosterApp:
    def __init__(self, seed=42, init_layers=8, init_wobble=0.35, init_palette=0, init_hearts=10):
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.layers = int(init_layers)
        self.wobble = float(init_wobble)
        self.palette_idx = int(init_palette)
        _, self.palette = get_palette(self.palette_idx)

        self.store = ShapeStore(
            base_blobs = make_default_blobs(self.layers, self.wobble, self.palette, self.rng),
            base_hearts = make_default_hearts(init_hearts, self.rng),
            added = []
        )

        # --- Figure ---
        self.fig = figure(
            x_range=(-4, 4), y_range=(-4, 4),
            match_aspect=True, width=880, height=880,
            tools="tap", toolbar_location=None, background_fill_color="#0d0f10"
        )
        self.fig.grid.visible = False
        self.fig.axis.visible = False

        # --- Controls ---
        self.title = pn.pane.Markdown("", sizing_mode="stretch_width")
        self.s_layers  = pn.widgets.IntSlider(name="Layers", start=6, end=12, step=1, value=self.layers)
        self.s_wobble  = pn.widgets.FloatSlider(name="Wobble", start=0.10, end=0.70, step=0.01, value=self.wobble)
        self.s_palette = pn.widgets.IntSlider(name="Palette", start=0, end=len(PALETTES)-1, step=1, value=self.palette_idx)

        self.b_mode  = pn.widgets.Button(name="Mode: Add HEART", button_type="primary")
        self.b_clear = pn.widgets.Button(name="Clear Added", button_type="warning", button_style="outline")

        self.mode = "heart"

        # --- Events ---
        self.s_layers.param.watch(self.on_layers_change, "value")
        self.s_wobble.param.watch(self.on_wobble_change, "value")
        self.s_palette.param.watch(self.on_palette_change, "value")
        self.b_clear.on_click(self.on_clear_added)
        self.b_mode.on_click(self.on_toggle_mode)
        self.fig.on_event(Tap, self.on_tap)

        # First render
        self.refresh()

        # --- Layout ---
        controls = pn.Column(
            pn.Row(self.s_layers, self.s_wobble, self.s_palette),
            pn.Row(self.b_mode, self.b_clear),
            sizing_mode="stretch_width"
        )
        self.view = pn.Column(self.title, self.fig, controls, sizing_mode="stretch_width")

    # ------------------ callbacks ------------------
    def all_shapes(self):
        blobs = list(self.store.base_blobs)
        hearts = list(self.store.base_hearts)
        for s in self.store.added:
            (blobs if s["type"]=="blob" else hearts).append(s)
        return blobs, hearts

    def on_layers_change(self, event):
        self.layers = int(event.new)
        _, pal = get_palette(self.palette_idx)
        self.store.base_blobs = make_default_blobs(self.layers, self.wobble, pal, self.rng)
        self.refresh()

    def on_wobble_change(self, event):
        self.wobble = float(event.new)
        for b in self.store.base_blobs:
            b["wobble"] = float(self.wobble * (0.9 + 0.25*np.random.rand()))
        self.refresh()

    def on_palette_change(self, event):
        self.palette_idx = int(event.new)
        _, pal = get_palette(self.palette_idx)
        for i, b in enumerate(self.store.base_blobs):
            b["color"] = pal[i % len(pal)]
        self.refresh()

    def on_clear_added(self, _):
        self.store.added = []
        self.refresh()

    def on_toggle_mode(self, _):
        self.mode = "blob" if self.mode == "heart" else "heart"
        self.b_mode.name = f"Mode: Add {self.mode.upper()}"
        # no re-render needed

    def on_tap(self, event: Tap):
        if event.x is None or event.y is None:
            return
        x, y = float(event.x), float(event.y)
        if self.mode == "blob":
            _, pal = get_palette(self.palette_idx)
            new_shape = {
                "type": "blob",
                "cx": x, "cy": y,
                "radius": float(self.rng.uniform(0.35, 1.15)),
                "wobble": float(self.wobble * self.rng.uniform(0.85, 1.25)),
                "color": pal[self.rng.integers(0, len(pal))],
                "points": int(self.rng.integers(200, 260)),
            }
        else:
            heart_palette = [
                (1.0, 0.4, 0.7, 1.0), (1.0, 0.2, 0.5, 1.0),
                (0.9, 0.3, 0.9, 1.0), (1.0, 0.85, 0.3, 1.0),
                (0.6, 0.9, 1.0, 1.0), (0.6, 1.0, 0.6, 1.0),
                (1.0, 1.0, 1.0, 1.0),
            ]
            new_shape = {
                "type": "heart",
                "cx": x, "cy": y,
                "scale": float(self.rng.uniform(0.05, 0.13)),
                "color": heart_palette[self.rng.integers(0, len(heart_palette))],
                "points": int(self.rng.integers(300, 360)),
            }
        self.store.added.append(new_shape)
        self.refresh()

    def refresh(self):
        name, _ = get_palette(self.palette_idx)
        blobs, hearts = self.all_shapes()
        self.title.object = f"**Palette:** {name} | **Mode:** {self.mode.upper()} | **Added:** {len(self.store.added)}"
        draw_scene(self.fig, blobs, hearts, add_grain=True)

# ---------- bootstrap ----------
pn.extension()
_app = PosterApp()
# Exportable/servable object for `panel serve app.py`
# Visiting /app will show the layout when using `panel serve app.py --show`
app = _app.view

if __name__ == "__main__":
    # You can also run: python app.py (this will call panel.serve)
    pn.serve({"app": app}, show=True)
