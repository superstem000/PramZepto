"""
Clean, presentation-quality versions of the four most useful scan plots:

    tilt_plane            - least-squares tilt plane of the focus surface
    z_predicted_scatter   - actual vs. model-predicted Z
    z_predicted_residuals - prediction residuals over scan order
    z_surface             - measured per-tile focus surface
    z_drift               - Z over time with model + mechanical-drift trend

Each is rendered individually, one graph per file, into plots/clean/.
Data loading / model logic is reused from characterization.py so these
stay consistent with the originals.

Usage:
    python plots/clean_plots.py
    python plots/clean_plots.py scan_log.csv --cycle 1 --output plots/clean
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the existing loaders / helpers from characterization.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from characterization import (  # noqa: E402
    load_scan_log,
    filter_return_phase,
    _build_z_map,
    _z_map_to_grid,
)

# ─── Shared style ────────────────────────────────────────────────

STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
}

ACCENT = "#1f6feb"     # actual / data
MODEL = "#d1495b"      # model / fit
TREND = "#2a9d8f"      # drift trend


def _fit_model(af):
    """Least-squares Z = a·col + b·row + c·tile + d  (same model as
    characterization.py). Returns (coeffs, predicted, residuals, rms)."""
    A = np.column_stack([
        af["col"].values,
        af["row"].values,
        af["tile_idx"].values,
        np.ones(len(af)),
    ])
    coeffs, *_ = np.linalg.lstsq(A, af["af_z"].values, rcond=None)
    pred = A @ coeffs
    resid = af["af_z"].values - pred
    rms = float(np.sqrt(np.mean(resid ** 2)))
    return coeffs, pred, resid, rms


def _save(fig, output_dir, name):
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")


# ─── 1. Tilt plane ───────────────────────────────────────────────

def plot_tilt_plane(df, output_dir):
    af = df[df["event"] == "periodic_af"].dropna(subset=["af_z"])
    if len(af) < 10:
        print("  tilt_plane: not enough periodic_af samples — skipped")
        return
    z_map = _build_z_map(df)
    if not z_map:
        print("  tilt_plane: no Z map — skipped")
        return
    grid, cols, rows = _z_map_to_grid(z_map)
    a, b, _c, d = _fit_model(af)[0]

    CC, RR = np.meshgrid(cols, rows)
    Z_plane = a * CC + b * RR + d
    grad = float(np.hypot(a, b))

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(
        Z_plane, origin="lower", aspect="equal", cmap="viridis",
        extent=[min(cols) - 0.5, max(cols) + 0.5,
                min(rows) - 0.5, max(rows) + 0.5],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fitted focus Z (fs)", fontsize=11)

    # Uphill gradient arrow, centred on the grid
    cx = 0.5 * (min(cols) + max(cols))
    cy = 0.5 * (min(rows) + max(rows))
    span = 0.28 * (max(cols) - min(cols) + 1)
    if grad > 0:
        ax.annotate(
            "", xy=(cx + a / grad * span, cy + b / grad * span),
            xytext=(cx, cy),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5,
                            mutation_scale=22),
        )
        ax.text(cx, cy - 0.10 * span, "uphill", color="white",
                ha="center", va="top", fontsize=10, fontweight="bold")

    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_title("Focus-Surface Tilt Plane")
    ax.grid(False)
    fig.text(
        0.5, -0.02,
        f"Least-squares plane:  Z = {a:.4f}·col + {b:.4f}·row + {d:.2f}   "
        f"|∇Z| = {grad:.4f} fs/step  (≈ {grad*30:.2f} fs across a 31-tile span)",
        ha="center", fontsize=10, color="#444",
    )
    _save(fig, output_dir, "tilt_plane")


# ─── 2. Actual vs predicted Z (scatter) ──────────────────────────

def plot_z_predicted_scatter(df, output_dir):
    af = df[df["event"] == "periodic_af"].dropna(subset=["af_z"])
    if len(af) < 4:
        print("  z_predicted_scatter: not enough samples — skipped")
        return
    (a, b, c, d), pred, resid, rms = _fit_model(af)
    actual = af["af_z"].values
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(pred, actual, s=22, alpha=0.45, c=ACCENT,
               edgecolors="none", label="AF samples")
    lo = min(pred.min(), actual.min())
    hi = max(pred.max(), actual.max())
    pad = 0.04 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            "--", color=MODEL, lw=1.8, label="ideal (y = x)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Predicted Z (fs)")
    ax.set_ylabel("Measured AF Z (fs)")
    ax.set_title("Measured vs. Predicted Focus Z")
    ax.legend(loc="upper left")
    ax.text(
        0.97, 0.05,
        f"R² = {r2:.4f}\nRMS = {rms:.3f} fs\nn = {len(af)}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc"),
    )
    fig.text(
        0.5, -0.02,
        f"Model:  Z = {a:.4f}·col + {b:.4f}·row + {c:.6f}·tile + {d:.2f}",
        ha="center", fontsize=10, color="#444",
    )
    _save(fig, output_dir, "z_predicted_scatter")


# ─── 3. Prediction residuals over scan order ─────────────────────

def plot_z_predicted_residuals(df, output_dir):
    af = df[df["event"] == "periodic_af"].dropna(subset=["af_z"])
    if len(af) < 4:
        print("  z_predicted_residuals: not enough samples — skipped")
        return
    _coeffs, _pred, resid, rms = _fit_model(af)
    tile = af["tile_idx"].values

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0, color=MODEL, lw=1.4, ls="--", zorder=1)
    ax.fill_between(tile, -rms, rms, color=TREND, alpha=0.15,
                    label=f"±1 RMS ({rms:.3f} fs)")
    ax.plot(tile, resid, ".-", color=ACCENT, ms=4, lw=0.6,
            alpha=0.8, label="residual")
    ax.set_xlabel("Tile index (scan order)")
    ax.set_ylabel("Measured − predicted Z (fs)")
    ax.set_title(f"Focus-Model Residuals   (RMS = {rms:.3f} fs)")
    ax.legend(loc="upper right", ncol=2)
    ax.margins(x=0.01)
    _save(fig, output_dir, "z_predicted_residuals")


# ─── 4. Measured Z surface ───────────────────────────────────────

def plot_z_surface(df, output_dir):
    z_map = _build_z_map(df)
    if not z_map:
        print("  z_surface: no Z map — skipped")
        return
    grid, cols, rows = _z_map_to_grid(z_map)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    # Masked (unvisited) cells fall through to the grey axes background —
    # avoids cmap.copy()/set_bad which need a newer matplotlib.
    ax.set_facecolor("#e8e8e8")
    im = ax.imshow(
        np.ma.masked_invalid(grid), origin="lower", aspect="equal",
        cmap="viridis", interpolation="nearest",
        extent=[min(cols) - 0.5, max(cols) + 0.5,
                min(rows) - 0.5, max(rows) + 0.5],
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Measured focus Z (fs)", fontsize=11)
    span = np.nanmax(grid) - np.nanmin(grid)
    ax.set_xlabel("Grid column")
    ax.set_ylabel("Grid row")
    ax.set_title("Measured Focus Surface")
    ax.grid(False)
    fig.text(
        0.5, -0.02,
        f"Per-tile autofocus Z over the scanned grid   "
        f"(range {span:.2f} fs; grey = not visited)",
        ha="center", fontsize=10, color="#444",
    )
    _save(fig, output_dir, "z_surface")


# ─── 5. Z drift over time ────────────────────────────────────────

def plot_z_drift(df, output_dir):
    af = df[df["event"] == "periodic_af"].dropna(subset=["af_z"]).copy()
    if af.empty:
        print("  z_drift: no periodic_af samples — skipped")
        return
    cap = df[df["event"] == "capture"].copy()
    (a, b, c, d), pred, resid, rms = _fit_model(af)

    tile = af["tile_idx"].values
    z = af["af_z"].values
    # Pure mechanical-drift trend (tilt averaged out)
    drift = c * tile + (z.mean() - c * tile.mean())
    per_cycle = c * 961.0

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1]},
    )

    ax = axes[0]
    if not cap.empty:
        ax.plot(cap["tile_idx"], cap["z_fs"], ".", color="#b8c4d8",
                ms=3, alpha=0.45, label="capture Z")
    ax.plot(tile, z, ".", color=ACCENT, ms=6, label="AF Z (ground truth)")
    ax.plot(tile, pred, "-", color=MODEL, lw=1.8, alpha=0.9,
            label="full model")
    ax.plot(tile, drift, "--", color=TREND, lw=2.0,
            label=f"drift trend: {c:.5f} fs/tile  ({per_cycle:.2f} fs/cycle)")
    ax.set_ylabel("Z (fs)")
    ax.set_title("Focus Z Over Scan Time")
    ax.legend(loc="upper right", ncol=2)

    axes[1].axhline(0, color=MODEL, lw=1.3, ls="--")
    axes[1].fill_between(tile, -rms, rms, color=TREND, alpha=0.15)
    axes[1].plot(tile, resid, ".-", color=ACCENT, ms=4, lw=0.6, alpha=0.8)
    axes[1].set_xlabel("Tile index (scan order)")
    axes[1].set_ylabel("Residual (fs)")
    axes[1].set_title(f"Model Residual   (RMS = {rms:.3f} fs)")
    axes[1].margins(x=0.01)

    fig.text(
        0.5, -0.01,
        f"Model:  Z = {a:.4f}·col + {b:.4f}·row + {c:.6f}·tile + {d:.2f}",
        ha="center", fontsize=10, color="#444",
    )
    fig.tight_layout()
    _save(fig, output_dir, "z_drift")


# ─── Driver ──────────────────────────────────────────────────────

PLOTS = [
    ("tilt_plane", plot_tilt_plane),
    ("z_predicted_scatter", plot_z_predicted_scatter),
    ("z_predicted_residuals", plot_z_predicted_residuals),
    ("z_surface", plot_z_surface),
    ("z_drift", plot_z_drift),
]


def main(argv):
    csv_path = os.path.join(_REPO_ROOT, "scan_log.csv")
    cycle = 1
    output_dir = os.path.join(_REPO_ROOT, "plots", "clean")

    i = 0
    args = argv[1:]
    positional = []
    while i < len(args):
        if args[i] == "--cycle" and i + 1 < len(args):
            cycle = int(args[i + 1]); i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_dir = args[i + 1]; i += 2
        else:
            positional.append(args[i]); i += 1
    if positional:
        csv_path = positional[0]

    print(f"Loading {csv_path}  (cycle {cycle})")
    df = load_scan_log(csv_path, cycle_filter=cycle)
    df = filter_return_phase(df)
    print(f"  {len(df)} rows after return-phase filter")

    os.makedirs(output_dir, exist_ok=True)
    with plt.rc_context(STYLE):
        for name, func in PLOTS:
            try:
                func(df, output_dir)
            except Exception as exc:  # keep going on a single bad plot
                print(f"  {name}: FAILED — {exc}")
                import traceback
                traceback.print_exc()
                plt.close("all")
    print(f"\nDone. Clean plots in {output_dir}")


if __name__ == "__main__":
    main(sys.argv)
