#!/usr/bin/env python3
"""Digitise the blue Sim curves of paper Fig. 4.3 from a 400 dpi page render.

Usage: digitize_fig43.py <page20.png> <out.csv>
Axes are found as the three black plot boxes; x is calibrated from the
60/70/80 mm tick marks under each box, y from the box edges (0 to 0.7 m/s,
calibrated against the 0.0/0.2/0.4/0.6 tick labels of panel A).
Digitisation error is dominated by line width: about +-0.005 m/s, +-0.15 mm.
"""
import sys
import numpy as np
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(int)
h, w, _ = img.shape
black = (img.sum(axis=2) < 200)
# Plot boxes: rows with long horizontal black runs in the figure band.
band = black[int(h*0.05):int(h*0.30), :]
rows = np.where(band.sum(axis=1) > 800)[0] + int(h*0.05)
top, bot = rows.min(), rows.max()
cols = np.where(black[top:top+3, :].sum(axis=0) >= 2)[0]
# cluster columns of the top edge into 3 panels
panels = []
start = cols[0]; prev = cols[0]
for c in cols[1:]:
    if c - prev > 20:
        panels.append((start, prev)); start = c
    prev = c
panels.append((start, prev))
panels = [p for p in panels if p[1]-p[0] > 300]
assert len(panels) == 3, panels
# tick calibration per panel: black ticks just below bottom edge
def ticks(x0, x1):
    strip = black[bot+2:bot+14, x0:x1+1]
    colsum = strip.sum(axis=0)
    xs = np.where(colsum > 6)[0] + x0
    groups = []
    s = xs[0]; p = xs[0]
    for c in xs[1:]:
        if c-p > 4: groups.append((s+p)//2); s = c
        p = c
    groups.append((s+p)//2)
    return groups
out = ["panel,specific_power_kw_per_m3,r_mm,sim_velocity_m_per_s"]
phis = [2.243, 4.038, 5.833]
blue = (img[:, :, 2] > 120) & (img[:, :, 0] < 110) & (img[:, :, 1] > 60) & (img[:, :, 1] < 180)
for pi, (x0, x1) in enumerate(panels):
    tk = ticks(x0, x1)
    assert len(tk) >= 3, (pi, tk)
    # ticks correspond to 60, 70, 80 mm (left to right, last three if extras)
    tk = tk[-3:]
    ax = np.polyfit(tk, [60.0, 70.0, 80.0], 1)
    ymap = lambda py: (bot - py) / (bot - top) * 0.7
    for px in range(x0+2, x1-1):
        if pi == 0 and px > x0 + 0.72*(x1-x0):
            continue  # legend box inside panel A
        col = np.where(blue[top+2:bot-1, px])[0]
        if len(col) == 0: continue
        py = col.mean() + top + 2
        r = np.polyval(ax, px)
        out.append(f"{'ABC'[pi]},{phis[pi]},{r:.2f},{ymap(py):.4f}")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print(f"wrote {len(out)-1} points, panels={panels}, top={top} bot={bot}")
