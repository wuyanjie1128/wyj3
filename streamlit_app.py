import streamlit as st
import numpy as np
import random
from typing import Tuple, List
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

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
        (1.00, 1.00, 1.0, 1.0),
    ]),
]

def rgba_to_css(rgba: Tuple[float, float, float, float]) -> str:
    # 提高不透明度，避免看起来偏暗
    r, g, b, _a = rgba
    return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.98)"

# Geometry helpers
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

def golden_angle_positions(n, radius_min=0.6, radius_max=2.6):
    phi = np.pi*(3 - np.sqrt(5))
    centers, radii = [], []
    for i in range(n):
        r = np.interp(i, [0, max(1, n-1)], [radius_min, radius_max])
        theta = i * phi
        cx, cy = 0.35*r*np.cos(theta), 0.35*r*np.sin(theta)
        centers.append((cx, cy)); radii.append(r)
    return centers, radii

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

# ---------- Streamlit state ----------
def init_state():
    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = 42
    if "rng" not in st.session_state:
        st.session_state.rng = np.random.default_rng(st.session_state.rng_seed)
    if "layers" not in st.session_state:
        st.session_state.layers = 8
    if "wobble" not in st.session_state:
        st.session_state.wobble = 0.35
    if "palette_idx" not in st.session_state:
        st.session_state.palette_idx = 0
    if "base_blobs" not in st.session_state:
        _, pal = PALETTES[st.session_state.palette_idx]
        st.session_state.base_blobs = make_default_blobs(st.session_state.layers, st.session_state.wobble, pal, st.session_state.rng)
    if "base_hearts" not in st.session_state:
        st.session_state.base_hearts = make_default_hearts(10, st.session_state.rng)
    if "added" not in st.session_state:
        st.session_state.added: List[dict] = []
    if "mode" not in st.session_state:
        st.session_state.mode = "heart"

init_state()
st.set_page_config(page_title="Poster App (Streamlit)", layout="wide")

# ---------- Sidebar controls ----------
st.sidebar.title("Controls")
layers = st.sidebar.slider("Layers", 6, 12, st.session_state.layers, 1)
wobble = st.sidebar.slider("Wobble", 0.10, 0.70, float(st.session_state.wobble), 0.01)
palette_idx = st.sidebar.selectbox(
    "Palette", list(range(len(PALETTES))), index=st.session_state.palette_idx,
    format_func=lambda i: PALETTES[i][0]
)

col1, col2, col3 = st.sidebar.columns(3)
toggle = col1.button(f"Mode: {'Add BLOB' if st.session_state.mode=='heart' else 'Add HEART'}")
clear = col2.button("Clear Added")
addrand = col3.button("Add Random")

if toggle:
    st.session_state.mode = "blob" if st.session_state.mode == "heart" else "heart"
if clear:
    st.session_state.added = []

# Apply control changes
changed = False
if layers != st.session_state.layers:
    st.session_state.layers = int(layers); changed = True
if abs(wobble - st.session_state.wobble) > 1e-9:
    st.session_state.wobble = float(wobble); changed = True
if palette_idx != st.session_state.palette_idx:
    st.session_state.palette_idx = int(palette_idx); changed = True
if changed:
    _, pal = PALETTES[st.session_state.palette_idx]
    st.session_state.base_blobs = make_default_blobs(st.session_state.layers, st.session_state.wobble, pal, st.session_state.rng)

# Random add button
if addrand:
    if st.session_state.mode == "blob":
        _, pal = PALETTES[st.session_state.palette_idx]
        st.session_state.added.append({
            "type": "blob",
            "cx": float(st.session_state.rng.uniform(-3.5, 3.5)),
            "cy": float(st.session_state.rng.uniform(-3.5, 3.5)),
            "radius": float(st.session_state.rng.uniform(0.35, 1.15)),
            "wobble": float(st.session_state.wobble * st.session_state.rng.uniform(0.85, 1.25)),
            "color": pal[st.session_state.rng.integers(0, len(pal))],
            "points": int(st.session_state.rng.integers(200, 260)),
        })
    else:
        heart_palette = [
            (1.0, 0.4, 0.7, 1.0), (1.0, 0.2, 0.5, 1.0),
            (0.9, 0.3, 0.9, 1.0), (1.0, 0.85, 0.3, 1.0),
            (0.6, 0.9, 1.0, 1.0), (0.6, 1.0, 0.6, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        ]
        st.session_state.added.append({
            "type": "heart",
            "cx": float(st.session_state.rng.uniform(-3.5, 3.5)),
            "cy": float(st.session_state.rng.uniform(-3.5, 3.5)),
            "scale": float(st.session_state.rng.uniform(0.05, 0.13)),
            "color": heart_palette[st.session_state.rng.integers(0, len(heart_palette))],
            "points": int(st.session_state.rng.integers(300, 360)),
        })

# ---------- Compose all shapes ----------
def all_shapes():
    blobs = list(st.session_state.base_blobs)
    hearts = list(st.session_state.base_hearts)
    for s in st.session_state.added:
        (blobs if s["type"]=="blob" else hearts).append(s)
    return blobs, hearts

# ---------- Plotly render (修复 selected/unselected 报错 & “全黑”问题) ----------
def render_plotly(blobs, hearts):
    fig = go.Figure()

    # 白色背景 + 固定比例 + 只发事件不做“选中/未选中”切换
    fig.update_layout(
        width=880, height=880,
        xaxis=dict(range=[-4,4], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-4,4], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        clickmode="event"  # 只发 click 事件，不触发选中态
    )

    # Blobs（不再传 line 到 selected/unselected，避免 ValueError）
    for b in blobs:
        x, y = make_blob(n_points=b["points"], radius=b["radius"], wobble=b["wobble"], irregularity=0.12)
        x, y = x + b["cx"], y + b["cy"]
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            fill="toself",
            line=dict(width=0),
            fillcolor=rgba_to_css(b["color"]),
            opacity=1.0,
            hoverinfo="skip",
            showlegend=False,
            # 只保留 marker（即使没 marker 也不会报错；不要放 line）
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=1.0)),
        ))

    # Hearts（同上）
    for h in hearts:
        hx, hy = make_heart(n_points=h["points"], scale=h["scale"], offset=(h["cx"], h["cy"]))
        fig.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="lines",
            fill="toself",
            line=dict(width=0),
            fillcolor=rgba_to_css(h["color"]),
            opacity=1.0,
            hoverinfo="skip",
            showlegend=False,
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=1.0)),
        ))

    # 点击捕获层（隐形网格，不改变任何选中状态）
    gx = np.linspace(-4, 4, 90)
    gy = np.linspace(-4, 4, 90)
    GX, GY = np.meshgrid(gx, gy)
    fig.add_trace(go.Scatter(
        x=GX.ravel(), y=GY.ravel(),
        mode="markers",
        marker=dict(size=12, opacity=0.001),
        hoverinfo="skip",
        showlegend=False,
        name="clickgrid"
    ))

    # 保险：清空潜在选中状态
    fig.update_traces(selectedpoints=None)
    return fig

# ---------- UI ----------
name, _ = PALETTES[st.session_state.palette_idx]
st.markdown(f"**Palette:** {name} | **Mode:** {st.session_state.mode.upper()} | **Added:** {len(st.session_state.added)}")

blobs, hearts = all_shapes()
fig = render_plotly(blobs, hearts)

# 捕获点击
events = plotly_events(fig, click_event=True, select_event=False, override_height=900, override_width=900)

if events:
    last = events[-1]
    cx, cy = float(last.get("x", 0.0)), float(last.get("y", 0.0))
    if st.session_state.mode == "blob":
        _, pal = PALETTES[st.session_state.palette_idx]
        st.session_state.added.append({
            "type": "blob",
            "cx": cx, "cy": cy,
            "radius": float(st.session_state.rng.uniform(0.35, 1.15)),
            "wobble": float(st.session_state.wobble * st.session_state.rng.uniform(0.85, 1.25)),
            "color": pal[st.session_state.rng.integers(0, len(pal))],
            "points": int(st.session_state.rng.integers(200, 260)),
        })
    else:
        heart_palette = [
            (1.0, 0.4, 0.7, 1.0), (1.0, 0.2, 0.5, 1.0),
            (0.9, 0.3, 0.9, 1.0), (1.0, 0.85, 0.3, 1.0),
            (0.6, 0.9, 1.0, 1.0), (0.6, 1.0, 0.6, 1.0),
            (1.0, 1.0, 1.0, 1.0),
        ]
        st.session_state.added.append({
            "type": "heart",
            "cx": cx, "cy": cy,
            "scale": float(st.session_state.rng.uniform(0.05, 0.13)),
            "color": heart_palette[st.session_state.rng.integers(0, len(heart_palette))],
            "points": int(st.session_state.rng.integers(300, 360)),
        })
    st.rerun()
