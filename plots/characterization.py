"""
Scan Log Visualizer v3

Usage:
    python visualize_scan.py scan_log.csv --cycle 1
    python visualize_scan.py scan_log.csv --cycle 1 --output plots/
"""

import sys, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def load_scan_log(path, cycle_filter=None):
    df = pd.read_csv(path)
    numeric_cols = ['cycle', 'tile_idx', 'col', 'row', 'x_fs', 'y_fs', 'z_fs',
                    'error_fs_x', 'error_fs_y', 'error_mag',
                    'dz_per_col', 'dz_per_row', 'dz_per_tile', 'af_z', 'af_score']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if cycle_filter is not None:
        df = df[df['cycle'] == cycle_filter]
    return df


def filter_return_phase(df):
    """Remove return-phase rows by detecting the pattern:
    after the spiral completes (961 tiles), tile_idx keeps incrementing
    but col/row trace a path back toward (0,0). We detect this as
    any rows where tile_idx > the max spiral tile count for that cycle,
    OR events explicitly tagged 'return*'.
    Also detect by: after the last tile in the spiral path (which ends at center),
    any subsequent rows in the same cycle are return phase.
    """
    # Method: for each cycle, find where the spiral ends.
    # The spiral visits 961 tiles (31x31). The spiral ends at center (15,15).
    # After that, col/row go back toward (0,0).
    # Detect: within each cycle, after we see the center tile (15,15) appear
    # AND col/row start decreasing, those are return rows.
    
    # Simpler approach: look for 'return' in event name
    mask_return = df['event'].str.contains('return', case=False, na=False)
    
    # Also detect by tile_idx > 961 within a cycle
    # (return tiles have tile_idx continuing from 961+)
    # But tile_idx might not reset per cycle. Use a different approach:
    # After the spiral's last tile (15,15), the path goes to (14,15), (13,15)...
    # The spiral ends at center. So within each cycle, once we see tiles
    # going from center back to (0,0), mark them.
    
    # Most robust: check if sequential rows show col or row DECREASING
    # toward 0 after having been at center. Group by cycle.
    for cycle in df['cycle'].dropna().unique():
        cycle_mask = df['cycle'] == cycle
        cycle_df = df[cycle_mask]
        
        # Find capture events in this cycle
        caps = cycle_df[cycle_df['event'] == 'capture']
        if len(caps) < 30:
            continue
        
        # Find where (15,15) first appears as a capture
        center_hits = caps[(caps['col'] == 15) & (caps['row'] == 15)]
        if center_hits.empty:
            continue
        
        # Everything after the first center capture in this cycle is return
        center_idx = center_hits.index[0]
        return_rows = cycle_df.index > center_idx
        mask_return = mask_return | (cycle_mask & return_rows)
    
    return df[~mask_return]


def _build_z_map(df):
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z'])
    cap = df[df['event'] == 'capture']
    z_map = {}
    for _, r in cap.iterrows():
        z_map[(int(r['col']), int(r['row']))] = r['z_fs']
    for _, r in af.iterrows():
        z_map[(int(r['col']), int(r['row']))] = r['af_z']
    return z_map


def _z_map_to_grid(z_map):
    cols = sorted(set(k[0] for k in z_map))
    rows = sorted(set(k[1] for k in z_map))
    grid = np.full((len(rows), len(cols)), np.nan)
    for (c, r), z in z_map.items():
        if c in cols and r in rows:
            grid[rows.index(r), cols.index(c)] = z
    return grid, cols, rows


# ─── Plot 1: Z Surface ───────────────────────────────────────────

def plot_z_surface(df, output_dir):
    z_map = _build_z_map(df)
    if not z_map: return
    grid, cols, rows = _z_map_to_grid(z_map)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(grid, origin='lower', aspect='equal',
                   extent=[min(cols)-0.5, max(cols)+0.5, min(rows)-0.5, max(rows)+0.5],
                   cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Z (full-steps)')
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    ax.set_title('Z Focus Surface')
    fig.savefig(os.path.join(output_dir, 'z_surface.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 2: Z Terrain 3D ────────────────────────────────────────

def plot_z_3d(df, output_dir):
    z_map = _build_z_map(df)
    if len(z_map) < 4: return
    grid, cols, rows = _z_map_to_grid(z_map)
    C, R = np.meshgrid(cols, rows)
    Z = grid.copy()
    mask = np.isnan(Z)
    if mask.any() and not mask.all():
        Z[mask] = np.nanmean(Z)
    for elev, azim, suffix in [(30, -60, ''), (60, -45, '_top'), (20, -120, '_side')]:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(C, R, Z, cmap='viridis', alpha=0.9, edgecolor='none',
                       antialiased=True, rstride=1, cstride=1)
        ax.set_xlabel('Column'); ax.set_ylabel('Row'); ax.set_zlabel('Z (full-steps)')
        ax.set_title('Z Terrain — Focus Surface')
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(os.path.join(output_dir, f'z_terrain_3d{suffix}.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)


# ─── Plot 3: Error Heatmap ───────────────────────────────────────

def plot_error_heatmap(df, output_dir):
    landings = df[df['event'] == 'landing'].dropna(subset=['error_mag'])
    if landings.empty: return
    cap_val = landings['error_mag'].quantile(0.95)
    err_map = landings.groupby(['col', 'row'])['error_mag'].mean()
    cols = sorted(landings['col'].unique().astype(int))
    rows = sorted(landings['row'].unique().astype(int))
    grid = np.full((len(rows), len(cols)), np.nan)
    for (c, r), e in err_map.items():
        grid[rows.index(int(r)), cols.index(int(c))] = min(e, cap_val)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(grid, origin='lower', aspect='equal',
                   extent=[min(cols)-0.5, max(cols)+0.5, min(rows)-0.5, max(rows)+0.5],
                   cmap='YlOrRd', vmin=0, vmax=cap_val)
    plt.colorbar(im, ax=ax, label='Error (full-steps)')
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    ax.set_title(f'Landing Error Magnitude (capped at {cap_val:.1f}fs)')
    fig.savefig(os.path.join(output_dir, 'error_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 4: Error Vectors (ALL, no outlier removal) ─────────────

def plot_error_vectors(df, output_dir):
    landings = df[df['event'] == 'landing'].dropna(subset=['error_fs_x', 'error_fs_y'])
    if landings.empty: return
    avg = landings.groupby(['col', 'row'])[['error_fs_x', 'error_fs_y']].mean()
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    for (c, r), row in avg.iterrows():
        mag = np.sqrt(row['error_fs_x']**2 + row['error_fs_y']**2)
        color = plt.cm.Reds(min(mag / 10.0, 1.0))  # scale color by magnitude
        ax.quiver(c, r, row['error_fs_x'], row['error_fs_y'],
                  angles='xy', scale_units='xy', scale=3, color=color, alpha=0.7, width=0.003)
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    ax.set_title('Landing Error Vectors (all tiles)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 31); ax.set_ylim(-1, 31)
    fig.savefig(os.path.join(output_dir, 'error_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 5: dz Convergence ──────────────────────────────────────

def plot_dz_convergence(df, output_dir):
    af = df[df['event'] == 'periodic_af'].copy()
    if af.empty: af = df[df['event'] == 'capture'].copy()
    if af.empty: return
    af = af.reset_index(drop=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, col, color, name in zip(axes, ['dz_per_col', 'dz_per_row', 'dz_per_tile'],
                                     ['b', 'r', 'g'], ['dz_per_col', 'dz_per_row', 'dz_per_tile']):
        ax.plot(af.index, af[col], f'{color}.-', markersize=2)
        ax.set_ylabel(name); ax.grid(True, alpha=0.3)
        ax.axhline(y=af[col].iloc[-1], color=color, linestyle='--', alpha=0.3)
    axes[2].set_xlabel('AF Sample Index')
    fig.suptitle('dz Parameter Convergence'); fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'dz_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 6: AF Scores ───────────────────────────────────────────

def plot_af_scores(df, output_dir):
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_score'])
    if af.empty: return
    score_map = af.groupby(['col', 'row'])['af_score'].mean()
    cols = sorted(af['col'].unique().astype(int))
    rows = sorted(af['row'].unique().astype(int))
    grid = np.full((len(rows), len(cols)), np.nan)
    for (c, r), s in score_map.items():
        grid[rows.index(int(r)), cols.index(int(c))] = s
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(grid, origin='lower', aspect='equal',
                   extent=[min(cols)-0.5, max(cols)+0.5, min(rows)-0.5, max(rows)+0.5], cmap='plasma')
    plt.colorbar(im, ax=ax, label='AF Score')
    ax.set_xlabel('Column'); ax.set_ylabel('Row'); ax.set_title('AF Score Across Grid')
    fig.savefig(os.path.join(output_dir, 'af_scores.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 7: Z Drift with trendlines ─────────────────────────────

def plot_z_drift(df, output_dir):
    cap = df[df['event'] == 'capture'].copy().reset_index(drop=True)
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z']).copy().reset_index(drop=True)
    
    if af.empty: return
    
    # Fit full model for trendlines
    A = np.column_stack([af['col'].values, af['row'].values,
                         af['tile_idx'].values, np.ones(len(af))])
    result, _, _, _ = np.linalg.lstsq(A, af['af_z'].values, rcond=None)
    fit_col, fit_row, fit_tile, fit_z0 = result
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: raw Z with model overlay
    ax = axes[0]
    if not cap.empty:
        ax.plot(cap['tile_idx'], cap['z_fs'], '.', color='#aabbdd', markersize=2, alpha=0.4, label='Capture Z')
    ax.plot(af['tile_idx'], af['af_z'], '.', color='#cc3333', markersize=4, label='AF Z (ground truth)')
    
    # Model prediction line
    z_model = A @ result
    ax.plot(af['tile_idx'], z_model, 'k-', linewidth=1.5, alpha=0.6,
            label=f'Model: {fit_col:.3f}×col + {fit_row:.3f}×row + {fit_tile:.5f}×tile + {fit_z0:.1f}')
    
    # Show the per-tile drift as a linear trend through the data
    tile_trend = fit_tile * af['tile_idx'].values + np.mean(af['af_z'].values) - fit_tile * np.mean(af['tile_idx'].values)
    ax.plot(af['tile_idx'], tile_trend, 'g--', linewidth=1, alpha=0.5,
            label=f'Drift trend: {fit_tile:.5f} fs/tile ({fit_tile*961:.2f} fs/cycle)')
    
    ax.set_ylabel('Z (full-steps)')
    ax.set_title('Z Position Over Time with Model Overlay')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)
    
    # Bottom: residuals (actual - model)
    residuals = af['af_z'].values - z_model
    axes[1].plot(af['tile_idx'], residuals, '.-', color='steelblue', markersize=3, linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].fill_between(af['tile_idx'], -0.5, 0.5, alpha=0.1, color='green', label='±0.5 fs')
    axes[1].set_xlabel('Tile Index')
    axes[1].set_ylabel('Residual (fs)')
    axes[1].set_title(f'Prediction Error (RMS = {np.sqrt(np.mean(residuals**2)):.3f} fs)')
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'z_drift.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 8: Z vs Predicted ──────────────────────────────────────

def plot_z_vs_predicted(df, output_dir):
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z'])
    if len(af) < 4: return
    A = np.column_stack([af['col'].values, af['row'].values, af['tile_idx'].values, np.ones(len(af))])
    result, _, _, _ = np.linalg.lstsq(A, af['af_z'].values, rcond=None)
    z_pred = A @ result; residuals = af['af_z'].values - z_pred
    rms = np.sqrt(np.mean(residuals**2))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(z_pred, af['af_z'].values, s=8, alpha=0.4, c='steelblue')
    lim = [min(z_pred.min(), af['af_z'].min()), max(z_pred.max(), af['af_z'].max())]
    axes[0].plot(lim, lim, 'r--'); axes[0].set_xlabel('Predicted Z'); axes[0].set_ylabel('Actual Z')
    axes[0].set_title('Actual vs Predicted'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(af['tile_idx'].values, residuals, '-', color='steelblue', linewidth=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Tile Index'); axes[1].set_ylabel('Residual (fs)')
    axes[1].set_title(f'Residuals (RMS={rms:.3f})'); axes[1].grid(True, alpha=0.3)
    r_cols = sorted(af['col'].unique().astype(int)); r_rows = sorted(af['row'].unique().astype(int))
    grid = np.full((len(r_rows), len(r_cols)), np.nan)
    for i, (_, row) in enumerate(af.iterrows()):
        c, r = int(row['col']), int(row['row'])
        if c in r_cols and r in r_rows: grid[r_rows.index(r), r_cols.index(c)] = residuals[i]
    vmax = np.nanmax(np.abs(grid))
    im = axes[2].imshow(grid, origin='lower', aspect='equal',
                        extent=[min(r_cols)-0.5, max(r_cols)+0.5, min(r_rows)-0.5, max(r_rows)+0.5],
                        cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[2], label='Residual (fs)')
    axes[2].set_xlabel('Column'); axes[2].set_ylabel('Row'); axes[2].set_title('Residual Map')
    fig.suptitle(f"Z: Actual vs Full Model\n"
                 f"Z = {result[0]:.4f}×col + {result[1]:.4f}×row + {result[2]:.6f}×tile + {result[3]:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'z_vs_predicted.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 9: Equation Visualization ──────────────────────────────

def plot_equation_viz(df, output_dir):
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z'])
    if len(af) < 10: return
    A = np.column_stack([af['col'].values, af['row'].values, af['tile_idx'].values, np.ones(len(af))])
    result, _, _, _ = np.linalg.lstsq(A, af['af_z'].values, rcond=None)
    fit_col, fit_row, fit_tile, fit_z0 = result
    z_pred = A @ result

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Z Prediction Model\n'
                 f'Z(col, row, tile) = {fit_col:.4f} × col + {fit_row:.4f} × row + '
                 f'{fit_tile:.6f} × tile + {fit_z0:.2f}', fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Component contributions
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(af['tile_idx'], fit_col * af['col'].values, 'b-', alpha=0.7, label=f'col ({fit_col:.4f} × col)')
    ax1.plot(af['tile_idx'], fit_row * af['row'].values, 'r-', alpha=0.7, label=f'row ({fit_row:.4f} × row)')
    ax1.plot(af['tile_idx'], fit_tile * af['tile_idx'].values, 'g-', alpha=0.7, label=f'tile ({fit_tile:.6f} × tile)')
    ax1.set_xlabel('Tile Index'); ax1.set_ylabel('Z contribution (fs)')
    ax1.set_title('Component Contributions'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2: Actual vs model
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(af['tile_idx'], af['af_z'].values, 'k.', markersize=2, alpha=0.3, label='Actual Z')
    ax2.plot(af['tile_idx'], z_pred, 'r-', linewidth=1, alpha=0.8, label='Model')
    ax2.fill_between(af['tile_idx'], z_pred - 1, z_pred + 1, alpha=0.1, color='red', label='±1 fs')
    ax2.set_xlabel('Tile Index'); ax2.set_ylabel('Z (fs)')
    ax2.set_title('Actual Z vs Model'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # Panel 3: Tilt plane
    ax3 = fig.add_subplot(2, 2, 3)
    CC, RR = np.meshgrid(np.arange(31), np.arange(31))
    Z_plane = fit_col * CC + fit_row * RR + fit_z0
    im = ax3.imshow(Z_plane, origin='lower', aspect='equal', extent=[-0.5, 30.5, -0.5, 30.5], cmap='viridis')
    plt.colorbar(im, ax=ax3, label='Z (fs)')
    tilt_mag = np.sqrt(fit_col**2 + fit_row**2)
    if tilt_mag > 0:
        ax3.annotate('', xy=(15 + fit_col/tilt_mag*8, 15 + fit_row/tilt_mag*8), xytext=(15, 15),
                    arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax3.set_xlabel('Column'); ax3.set_ylabel('Row')
    ax3.set_title(f'Tilt Plane\ncol={fit_col:.4f} fs/step, row={fit_row:.4f} fs/step')

    # Panel 4: Drift isolated
    ax4 = fig.add_subplot(2, 2, 4)
    z_detrended = af['af_z'].values - (fit_col * af['col'].values + fit_row * af['row'].values + fit_z0)
    z_drift_model = fit_tile * af['tile_idx'].values
    ax4.plot(af['tile_idx'], z_detrended, 'k.', markersize=2, alpha=0.3, label='Actual (tilt removed)')
    ax4.plot(af['tile_idx'], z_drift_model, 'g-', linewidth=2, label=f'Drift: {fit_tile:.6f} × tile')
    ax4.set_xlabel('Tile Index'); ax4.set_ylabel('Z residual (fs)')
    ax4.set_title(f'Mechanical Drift: {fit_tile:.6f} fs/tile = {fit_tile*961:.2f} fs/cycle')
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'equation_viz.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 10: Tilt Decomposition ─────────────────────────────────

def plot_tilt_decomposition(df, output_dir):
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z'])
    if len(af) < 10: return
    z_map = _build_z_map(df)
    grid, cols, rows = _z_map_to_grid(z_map)
    
    # Full fit with tile
    A = np.column_stack([af['col'].values, af['row'].values, af['tile_idx'].values, np.ones(len(af))])
    result, _, _, _ = np.linalg.lstsq(A, af['af_z'].values, rcond=None)
    fit_col, fit_row, fit_tile, fit_z0 = result
    
    # Build predicted grid (tile term averaged per position from the actual tile indices)
    CC, RR = np.meshgrid(cols, rows)
    Z_plane = fit_col * CC + fit_row * RR + fit_z0
    
    # For each (col,row), find the average tile_idx when it was visited
    cap = df[df['event'] == 'capture']
    tile_idx_map = {}
    for _, r in cap.iterrows():
        tile_idx_map[(int(r['col']), int(r['row']))] = r['tile_idx']
    tile_grid = np.full_like(grid, np.nan)
    for (c, r), t in tile_idx_map.items():
        if c in cols and r in rows:
            tile_grid[rows.index(r), cols.index(c)] = t
    
    Z_full_pred = Z_plane + fit_tile * np.where(np.isnan(tile_grid), 0, tile_grid)
    Z_residual = grid - Z_full_pred
    
    kwargs = dict(origin='lower', aspect='equal',
                  extent=[min(cols)-0.5, max(cols)+0.5, min(rows)-0.5, max(rows)+0.5])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    im0 = axes[0,0].imshow(grid, cmap='viridis', **kwargs)
    plt.colorbar(im0, ax=axes[0,0], label='Z (fs)')
    axes[0,0].set_title('Actual Z'); axes[0,0].set_xlabel('Column'); axes[0,0].set_ylabel('Row')
    
    im1 = axes[0,1].imshow(Z_plane, cmap='viridis', vmin=np.nanmin(grid), vmax=np.nanmax(grid), **kwargs)
    plt.colorbar(im1, ax=axes[0,1], label='Z (fs)')
    axes[0,1].set_title(f'Tilt Plane Only\ndz_col={fit_col:.4f}, dz_row={fit_row:.4f}')
    axes[0,1].set_xlabel('Column'); axes[0,1].set_ylabel('Row')
    
    # Show the drift component spatially
    Z_drift = fit_tile * np.where(np.isnan(tile_grid), np.nan, tile_grid)
    im2 = axes[1,0].imshow(Z_drift, cmap='RdYlBu_r', **kwargs)
    plt.colorbar(im2, ax=axes[1,0], label='Z drift (fs)')
    axes[1,0].set_title(f'Temporal Drift Component\n{fit_tile:.6f} fs/tile (spiral order imprint)')
    axes[1,0].set_xlabel('Column'); axes[1,0].set_ylabel('Row')
    
    vmax_r = np.nanmax(np.abs(Z_residual))
    if vmax_r == 0: vmax_r = 1
    im3 = axes[1,1].imshow(Z_residual, cmap='RdBu_r', vmin=-vmax_r, vmax=vmax_r, **kwargs)
    plt.colorbar(im3, ax=axes[1,1], label='Residual (fs)')
    axes[1,1].set_title(f'Residual (Actual − Full Model)\nRMS={np.sqrt(np.nanmean(Z_residual**2)):.2f} fs')
    axes[1,1].set_xlabel('Column'); axes[1,1].set_ylabel('Row')
    
    fig.suptitle(f'Z Decomposition: Actual = Tilt + Drift + Residual\n'
                 f'Z = {fit_col:.4f}×col + {fit_row:.4f}×row + {fit_tile:.6f}×tile + {fit_z0:.2f}', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'tilt_decomposition.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 11: Z by Direction — show tilt from raw data ───────────

def plot_z_by_direction(df, output_dir):
    """Show how Z varies along each axis independently, with tile drift removed."""
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z'])
    if len(af) < 20: return
    
    # First fit full model to get tile drift
    A_full = np.column_stack([af['col'].values, af['row'].values,
                              af['tile_idx'].values, np.ones(len(af))])
    full_result, _, _, _ = np.linalg.lstsq(A_full, af['af_z'].values, rcond=None)
    fit_col, fit_row, fit_tile, fit_z0 = full_result
    
    # Remove tile drift from Z to show pure spatial tilt
    z_no_drift = af['af_z'].values - fit_tile * af['tile_idx'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Z (drift-removed) vs Column, colored by row
    ax = axes[0, 0]
    scatter = ax.scatter(af['col'], z_no_drift, c=af['row'], s=8, alpha=0.5, cmap='coolwarm')
    plt.colorbar(scatter, ax=ax, label='Row')
    for fixed_row in [0, 10, 20, 30]:
        mask = abs(af['row'] - fixed_row) <= 1
        subset_col = af['col'].values[mask]
        subset_z = z_no_drift[mask]
        if len(subset_col) > 3:
            p = np.polyfit(subset_col, subset_z, 1)
            x_fit = np.linspace(subset_col.min(), subset_col.max(), 50)
            ax.plot(x_fit, np.polyval(p, x_fit), 'k-', linewidth=1.5, alpha=0.5)
            ax.text(x_fit[-1], np.polyval(p, x_fit[-1]), f'row≈{fixed_row}\nslope={p[0]:.3f}',
                   fontsize=7, ha='left')
    ax.set_xlabel('Column'); ax.set_ylabel('Z (drift-removed, fs)')
    ax.set_title('Z vs Column (drift removed, colored by row)\nSlopes ≈ dz_per_col')
    ax.grid(True, alpha=0.3)
    
    # Top-right: Z (drift-removed) vs Row, colored by column
    ax = axes[0, 1]
    scatter = ax.scatter(af['row'], z_no_drift, c=af['col'], s=8, alpha=0.5, cmap='coolwarm')
    plt.colorbar(scatter, ax=ax, label='Column')
    for fixed_col in [0, 10, 20, 30]:
        mask = abs(af['col'] - fixed_col) <= 1
        subset_row = af['row'].values[mask]
        subset_z = z_no_drift[mask]
        if len(subset_row) > 3:
            p = np.polyfit(subset_row, subset_z, 1)
            x_fit = np.linspace(subset_row.min(), subset_row.max(), 50)
            ax.plot(x_fit, np.polyval(p, x_fit), 'k-', linewidth=1.5, alpha=0.5)
            ax.text(x_fit[-1], np.polyval(p, x_fit[-1]), f'col≈{fixed_col}\nslope={p[0]:.3f}',
                   fontsize=7, ha='left')
    ax.set_xlabel('Row'); ax.set_ylabel('Z (drift-removed, fs)')
    ax.set_title('Z vs Row (drift removed, colored by column)\nSlopes ≈ dz_per_row')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Pure drift (tilt removed)
    A_tilt = np.column_stack([af['col'].values, af['row'].values, np.ones(len(af))])
    tilt_result, _, _, _ = np.linalg.lstsq(A_tilt, af['af_z'].values, rcond=None)
    z_no_tilt = af['af_z'].values - A_tilt @ tilt_result
    ax = axes[1, 0]
    ax.plot(af['tile_idx'], z_no_tilt, '.', color='steelblue', markersize=3, alpha=0.5)
    p_drift = np.polyfit(af['tile_idx'].values, z_no_tilt, 1)
    x_fit = np.linspace(af['tile_idx'].min(), af['tile_idx'].max(), 100)
    ax.plot(x_fit, np.polyval(p_drift, x_fit), 'g-', linewidth=2,
            label=f'Drift: {p_drift[0]:.6f} fs/tile')
    ax.set_xlabel('Tile Index'); ax.set_ylabel('Z − tilt (fs)')
    ax.set_title('Mechanical Drift (tilt removed)')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # Bottom-right: Z colored by spiral ring
    ax = axes[1, 1]
    rings = np.minimum(np.minimum(af['col'], af['row']),
                       np.minimum(30 - af['col'], 30 - af['row']))
    scatter = ax.scatter(af['tile_idx'], af['af_z'], c=rings, s=8, alpha=0.6, cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Spiral ring (0=edge, 15=center)')
    ax.set_xlabel('Tile Index'); ax.set_ylabel('Z (fs)')
    ax.set_title('Z Over Time colored by spiral ring\nOscillation = tilt, trend = drift')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Z Behavior: Tilt and Drift from Raw Data', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'z_by_direction.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Plot 12: Summary ────────────────────────────────────────────

def plot_summary(df, output_dir):
    af = df[df['event'] == 'periodic_af'].dropna(subset=['af_z'])
    landings = df[df['event'] == 'landing'].dropna(subset=['error_mag'])
    cap = df[df['event'] == 'capture']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    if not landings.empty:
        axes[0,0].hist(landings['error_mag'].clip(upper=landings['error_mag'].quantile(0.95)),
                       bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
        axes[0,0].axvline(x=2.0, color='red', linestyle='--', label='Correction threshold')
        axes[0,0].set_xlabel('Landing Error (fs)'); axes[0,0].set_ylabel('Count')
        axes[0,0].set_title(f'Error Distribution — Median: {landings["error_mag"].median():.2f}fs')
        axes[0,0].legend()
    if not af.empty:
        axes[0,1].hist(af['af_score'], bins=30, color='orange', edgecolor='white', linewidth=0.5)
        axes[0,1].set_xlabel('AF Score'); axes[0,1].set_ylabel('Count')
        axes[0,1].set_title(f'AF Scores — Median: {af["af_score"].median():.1f}')
    if not af.empty:
        row_z = af.groupby('row')['af_z'].mean()
        axes[1,0].plot(row_z.index, row_z.values, 'r.-')
        axes[1,0].set_xlabel('Row'); axes[1,0].set_ylabel('Mean Z (fs)')
        axes[1,0].set_title('Mean Z per Row'); axes[1,0].grid(True, alpha=0.3)
    if not af.empty:
        col_z = af.groupby('col')['af_z'].mean()
        axes[1,1].plot(col_z.index, col_z.values, 'b.-')
        axes[1,1].set_xlabel('Column'); axes[1,1].set_ylabel('Mean Z (fs)')
        axes[1,1].set_title('Mean Z per Column'); axes[1,1].grid(True, alpha=0.3)
    fig.suptitle(f'Scan Summary — {len(cap)} tiles, {len(af)} AF samples')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────

def generate_all_plots(csv_path, output_dir=None, cycle_filter=None):
    print(f"Loading {csv_path}...")
    df = load_scan_log(csv_path, cycle_filter=cycle_filter)
    df = filter_return_phase(df)
    print(f"  {len(df)} rows (return phase filtered)")
    print(f"  Events: {df['event'].value_counts().to_dict()}")
    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "plots")
    os.makedirs(output_dir, exist_ok=True)
    for name, func in [("z_surface", plot_z_surface), ("z_terrain_3d", plot_z_3d),
                        ("error_heatmap", plot_error_heatmap), ("error_vectors", plot_error_vectors),
                        ("dz_convergence", plot_dz_convergence), ("af_scores", plot_af_scores),
                        ("z_drift", plot_z_drift), ("z_vs_predicted", plot_z_vs_predicted),
                        ("equation_viz", plot_equation_viz), ("tilt_decomposition", plot_tilt_decomposition),
                        ("z_by_direction", plot_z_by_direction), ("summary", plot_summary)]:
        try:
            print(f"  {name}..."); func(df, output_dir); print(f"    Done")
        except Exception as e:
            print(f"    Skipped {name}: {e}"); import traceback; traceback.print_exc(); plt.close('all')
    
    # Generate interactive HTML
    try:
        print(f"  z_terrain_interactive...")
        generate_interactive_3d(csv_path, output_dir, cycle_filter=cycle_filter)
        print(f"    Done")
    except Exception as e:
        print(f"    Skipped interactive 3D: {e}")
    
    print(f"\nAll plots saved to {output_dir}")


def generate_interactive_3d(csv_path, output_dir=None, cycle_filter=None):
    """Generate an interactive 3D HTML terrain using Plotly."""
    df = load_scan_log(csv_path, cycle_filter=cycle_filter)
    df = filter_return_phase(df)
    z_map = _build_z_map(df)
    if len(z_map) < 4:
        print("Not enough data for 3D terrain")
        return
    grid, cols, rows = _z_map_to_grid(z_map)
    
    # Fill NaN with mean for surface
    Z = grid.copy()
    mask = np.isnan(Z)
    if mask.any() and not mask.all():
        Z[mask] = np.nanmean(Z)
    
    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to JSON-safe lists
    z_data = Z.tolist()
    col_data = [int(c) for c in cols]
    row_data = [int(r) for r in rows]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Z Terrain - Interactive 3D</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; background: #1a1a2e; font-family: 'Segoe UI', sans-serif; }}
        #plot {{ width: 100vw; height: 100vh; }}
        .controls {{
            position: fixed; top: 16px; right: 16px; z-index: 100;
            background: rgba(26,26,46,0.9); padding: 16px; border-radius: 8px;
            color: #e0e0e0; font-size: 13px; backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .controls label {{ display: block; margin: 8px 0 4px; color: #aaa; }}
        .controls select, .controls input {{
            background: #2a2a4a; color: #fff; border: 1px solid #444;
            padding: 4px 8px; border-radius: 4px; width: 100%;
        }}
        h3 {{ margin: 0 0 8px; color: #7eb8da; font-size: 14px; }}
    </style>
</head>
<body>
    <div id="plot"></div>
    <div class="controls">
        <h3>Z Terrain Controls</h3>
        <label>Color Scale</label>
        <select id="colorscale" onchange="updatePlot()">
            <option value="Viridis" selected>Viridis</option>
            <option value="Plasma">Plasma</option>
            <option value="Inferno">Inferno</option>
            <option value="Turbo">Turbo</option>
            <option value="Earth">Earth</option>
            <option value="Hot">Hot</option>
        </select>
        <label>Z Exaggeration</label>
        <input type="range" id="zscale" min="0.5" max="5" step="0.1" value="1" oninput="updatePlot()">
        <span id="zval">1.0x</span>
    </div>
    <script>
        const cols = {col_data};
        const rows = {row_data};
        const z = {z_data};
        const zMin = Math.min(...z.flat());
        const zMax = Math.max(...z.flat());

        function updatePlot() {{
            const cs = document.getElementById('colorscale').value;
            const zs = parseFloat(document.getElementById('zscale').value);
            document.getElementById('zval').textContent = zs.toFixed(1) + 'x';

            const scaledZ = z.map(row => row.map(v => (v - zMin) * zs + zMin));

            const data = [{{
                type: 'surface',
                x: cols,
                y: rows,
                z: scaledZ,
                colorscale: cs,
                colorbar: {{ title: 'Z (fs)', titleside: 'right' }},
                hovertemplate: 'Col: %{{x}}<br>Row: %{{y}}<br>Z: %{{customdata:.2f}} fs<extra></extra>',
                customdata: z,
                lighting: {{ ambient: 0.6, diffuse: 0.7, specular: 0.3, roughness: 0.5 }},
                contours: {{
                    z: {{ show: true, usecolormap: true, highlightcolor: '#fff', project: {{ z: true }} }}
                }}
            }}];

            const layout = {{
                scene: {{
                    xaxis: {{ title: 'Column', gridcolor: '#333', color: '#aaa' }},
                    yaxis: {{ title: 'Row', gridcolor: '#333', color: '#aaa' }},
                    zaxis: {{ title: 'Z (full-steps)', gridcolor: '#333', color: '#aaa' }},
                    bgcolor: '#1a1a2e',
                    camera: {{ eye: {{ x: 1.5, y: -1.5, z: 1.2 }} }}
                }},
                paper_bgcolor: '#1a1a2e',
                margin: {{ l: 0, r: 0, t: 40, b: 0 }},
                title: {{
                    text: 'Z Focus Terrain — Interactive',
                    font: {{ color: '#7eb8da', size: 16 }}
                }}
            }};

            Plotly.react('plot', data, layout, {{ responsive: true }});
        }}

        updatePlot();
    </script>
</body>
</html>"""
    
    out_path = os.path.join(output_dir, 'z_terrain_interactive.html')
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"Interactive 3D terrain saved to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_scan.py <scan_log.csv> [--cycle N] [--output dir]"); sys.exit(1)
    csv_path = sys.argv[1]; output_dir = None; cycle_filter = None; i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--cycle' and i+1 < len(sys.argv): cycle_filter = int(sys.argv[i+1]); i += 2
        elif sys.argv[i] == '--output' and i+1 < len(sys.argv): output_dir = sys.argv[i+1]; i += 2
        else: output_dir = sys.argv[i]; i += 1
    files = glob.glob(csv_path)
    if not files: print(f"No files found: {csv_path}"); sys.exit(1)
    for f in sorted(files): generate_all_plots(f, output_dir, cycle_filter=cycle_filter)