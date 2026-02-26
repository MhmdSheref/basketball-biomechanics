"""
Circle fitting on RWrist relative trajectory.
Uses algebraic least-squares circle fit (Kåsa method).
Segments:
  1) 1.085s to 1.974s
  2) 2.627s to end
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

# ── Load data ──
time_arr, rel_x, rel_y = [], [], []
with open("RWrist_Relative_Position.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if len(row) == 3:
            time_arr.append(float(row[0]))
            rel_x.append(float(row[1]))
            rel_y.append(float(row[2]))

time_arr = np.array(time_arr)
rel_x = np.array(rel_x)
rel_y = np.array(rel_y)

# ── Circle fit (Kåsa algebraic method) ──
def fit_circle(x, y):
    """Fit a circle to 2D points using least-squares (Kåsa method).
    Returns (cx, cy, r) — center and radius."""
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = result[0], result[1]
    r = np.sqrt(result[2] + cx**2 + cy**2)
    return cx, cy, r

# ── Segment 1: 1.085s to 1.974s ──
mask1 = (time_arr >= 1.085) & (time_arr <= 1.974)
x1, y1 = rel_x[mask1], rel_y[mask1]
cx1, cy1, r1 = fit_circle(x1, y1)
print(f"Segment 1 (1.085s – 1.974s): {mask1.sum()} points")
print(f"  Center: ({cx1:.6f}, {cy1:.6f}) m")
print(f"  Radius: {r1:.6f} m  ({r1*100:.2f} cm)")

# ── Segment 2: 2.627s to end ──
mask2 = time_arr >= 2.627
x2, y2 = rel_x[mask2], rel_y[mask2]
cx2, cy2, r2 = fit_circle(x2, y2)
print(f"\nSegment 2 (2.627s – end): {mask2.sum()} points")
print(f"  Center: ({cx2:.6f}, {cy2:.6f}) m")
print(f"  Radius: {r2:.6f} m  ({r2*100:.2f} cm)")

# ── Residual analysis ──
dist1 = np.sqrt((x1 - cx1)**2 + (y1 - cy1)**2) - r1
dist2 = np.sqrt((x2 - cx2)**2 + (y2 - cy2)**2) - r2
print(f"\nSegment 1 residuals: mean={np.mean(np.abs(dist1))*100:.3f} cm, max={np.max(np.abs(dist1))*100:.3f} cm")
print(f"Segment 2 residuals: mean={np.mean(np.abs(dist2))*100:.3f} cm, max={np.max(np.abs(dist2))*100:.3f} cm")

# ── Plot ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Full trajectory with both segments highlighted
ax0 = axes[0]
ax0.plot(rel_x, rel_y, color="#bbb", linewidth=0.8, label="Full trajectory")
ax0.plot(x1, y1, color="#2563eb", linewidth=2, label=f"Segment 1 (R={r1*100:.1f} cm)")
ax0.plot(x2, y2, color="#dc2626", linewidth=2, label=f"Segment 2 (R={r2*100:.1f} cm)")
theta = np.linspace(0, 2*np.pi, 200)
ax0.plot(cx1 + r1*np.cos(theta), cy1 + r1*np.sin(theta), '--', color="#2563eb", alpha=0.7, linewidth=1.5)
ax0.plot(cx2 + r2*np.cos(theta), cy2 + r2*np.sin(theta), '--', color="#dc2626", alpha=0.7, linewidth=1.5)
ax0.plot(cx1, cy1, 'x', color="#2563eb", markersize=10, markeredgewidth=2)
ax0.plot(cx2, cy2, 'x', color="#dc2626", markersize=10, markeredgewidth=2)
ax0.set_xlabel("Rel X (m)")
ax0.set_ylabel("Rel Y (m)")
ax0.set_title("RWrist Relative Position — Circle Fits")
ax0.legend(fontsize=9)
ax0.set_aspect("equal")
ax0.grid(True, alpha=0.3)

# Plot 2: Segment 1 zoomed
ax1p = axes[1]
ax1p.plot(x1, y1, 'o', color="#2563eb", markersize=2, alpha=0.6)
ax1p.plot(cx1 + r1*np.cos(theta), cy1 + r1*np.sin(theta), '-', color="#2563eb", linewidth=2)
ax1p.plot(cx1, cy1, 'x', color="#2563eb", markersize=12, markeredgewidth=3)
ax1p.set_xlabel("Rel X (m)")
ax1p.set_ylabel("Rel Y (m)")
ax1p.set_title(f"Segment 1: 1.085s – 1.974s\nR = {r1*100:.2f} cm")
ax1p.set_aspect("equal")
ax1p.grid(True, alpha=0.3)

# Plot 3: Segment 2 zoomed
ax2p = axes[2]
ax2p.plot(x2, y2, 'o', color="#dc2626", markersize=2, alpha=0.6)
ax2p.plot(cx2 + r2*np.cos(theta), cy2 + r2*np.sin(theta), '-', color="#dc2626", linewidth=2)
ax2p.plot(cx2, cy2, 'x', color="#dc2626", markersize=12, markeredgewidth=3)
ax2p.set_xlabel("Rel X (m)")
ax2p.set_ylabel("Rel Y (m)")
ax2p.set_title(f"Segment 2: 2.627s – end\nR = {r2*100:.2f} cm")
ax2p.set_aspect("equal")
ax2p.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("circle_fit_results.png", dpi=150, bbox_inches="tight")
print("\nSaved: circle_fit_results.png")
