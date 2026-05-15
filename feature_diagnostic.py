"""
Feature Analysis Diagnostic Tool

Run on a captured image to extract statistics about dark features.
Outputs contour data, intensity histograms, and an annotated image.

Usage:
    python3 feature_diagnostic.py /path/to/captured_image.jpg

Outputs (saved next to input image):
    - *_annotated.jpg   — contours drawn with area labels
    - *_stats.txt       — detailed statistics
"""

import sys
import os
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    if HAS_PIL:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    else:
        raise ImportError("PIL/Pillow required: pip install Pillow")


def get_analysis_channel(rgb: np.ndarray) -> np.ndarray:
    """
    Pick the channel with the most dynamic range for analysis.
    For red-filtered images, this returns the red channel directly.
    """
    ranges = []
    for i, name in enumerate(["Red", "Green", "Blue"]):
        ch = rgb[:, :, i]
        r = float(ch.max()) - float(ch.min())
        ranges.append((r, i, name))
    
    ranges.sort(reverse=True)
    best_range, best_idx, best_name = ranges[0]
    print(f"  Using {best_name} channel for analysis (range={best_range:.0f})")
    return rgb[:, :, best_idx], best_name


def find_contours_simple(binary: np.ndarray):
    """
    Simple contour/connected-component finder using flood fill.
    Returns list of dicts with contour stats.
    No OpenCV dependency.
    """
    labeled = np.zeros_like(binary, dtype=np.int32)
    label = 0
    h, w = binary.shape

    contours = []

    for y in range(h):
        for x in range(w):
            if binary[y, x] == 255 and labeled[y, x] == 0:
                # BFS flood fill
                label += 1
                queue = [(y, x)]
                pixels = []
                labeled[y, x] = label

                while queue:
                    cy, cx = queue.pop(0)
                    pixels.append((cy, cx))

                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary[ny, nx] == 255 and labeled[ny, nx] == 0:
                                labeled[ny, nx] = label
                                queue.append((ny, nx))

                if len(pixels) < 5:
                    continue  # skip tiny noise

                ys = [p[0] for p in pixels]
                xs = [p[1] for p in pixels]

                contours.append({
                    'label': label,
                    'area': len(pixels),
                    'y_min': min(ys),
                    'y_max': max(ys),
                    'x_min': min(xs),
                    'x_max': max(xs),
                    'centroid_y': np.mean(ys),
                    'centroid_x': np.mean(xs),
                    'bbox_w': max(xs) - min(xs) + 1,
                    'bbox_h': max(ys) - min(ys) + 1,
                })

    return contours, labeled


def analyze_image(path: str):
    """Main analysis function."""
    print(f"\n{'='*60}")
    print(f"Feature Diagnostic: {path}")
    print(f"{'='*60}\n")

    rgb = load_image(path)
    gray_lum = (0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]).astype(np.uint8)
    h, w = rgb.shape[:2]
    print(f"Image size: {w} x {h}")
    print(f"Channels: {rgb.shape[2] if rgb.ndim == 3 else 1}")

    # Per-channel stats
    print(f"\n--- Per-Channel Stats ---")
    for i, name in enumerate(["Red", "Green", "Blue"]):
        ch = rgb[:, :, i]
        print(f"  {name:6s}: min={ch.min():3d}  max={ch.max():3d}  "
              f"mean={ch.mean():.1f}  std={ch.std():.1f}")

    # Pick the best channel for analysis (red for red-filtered images)
    print()
    gray, ch_name = get_analysis_channel(rgb)

    # ── Intensity stats on analysis channel ──
    print(f"\n--- {ch_name} Channel Intensity ---")
    print(f"  Min:    {gray.min()}")
    print(f"  Max:    {gray.max()}")
    print(f"  Mean:   {gray.mean():.1f}")
    print(f"  Median: {np.median(gray):.1f}")
    print(f"  Std:    {gray.std():.1f}")

    # ── Try multiple thresholds ──
    print(f"\n--- Threshold Analysis ---")
    print(f"  (Testing which threshold isolates features best)\n")

    # Compute histogram for guidance
    hist, bin_edges = np.histogram(gray.ravel(), bins=50)
    peak_bin = np.argmax(hist)
    background_level = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
    print(f"  Background peak intensity: ~{background_level:.0f}")

    results_by_thresh = {}

    for thresh in [30, 50, 70, 90, 110, 130, 150]:
        binary = np.where(gray < thresh, 255, 0).astype(np.uint8)
        dark_pct = np.sum(binary == 255) / (h * w) * 100

        # Quick connected-component count (simplified)
        # Just count dark pixel percentage for speed
        results_by_thresh[thresh] = dark_pct
        print(f"  Threshold < {thresh:3d}: {dark_pct:.2f}% dark pixels")

    # ── Full contour analysis at a reasonable threshold ──
    # Pick threshold where dark pixels are between 0.5% and 10%
    best_thresh = 90  # default
    for t in [50, 70, 90, 110, 130]:
        if 0.1 < results_by_thresh.get(t, 0) < 15.0:
            best_thresh = t
            break
    
    best_thresh = 80 #use as default

    print(f"\n--- Contour Analysis (threshold < {best_thresh}) ---")
    binary = np.where(gray < best_thresh, 255, 0).astype(np.uint8)

    # Use scipy if available (much faster), otherwise pure numpy
    try:
        from scipy import ndimage
        labeled_array, num_features = ndimage.label(binary)
        print(f"  Using scipy.ndimage for labeling (fast)")

        contours = []
        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled_array == i)
            area = len(ys)
            if area < 5:
                continue
            contours.append({
                'label': i,
                'area': area,
                'y_min': int(ys.min()),
                'y_max': int(ys.max()),
                'x_min': int(xs.min()),
                'x_max': int(xs.max()),
                'centroid_y': float(ys.mean()),
                'centroid_x': float(xs.mean()),
                'bbox_w': int(xs.max() - xs.min() + 1),
                'bbox_h': int(ys.max() - ys.min() + 1),
            })
    except ImportError:
        print(f"  scipy not available, using pure numpy flood fill (slower)...")
        contours, _ = find_contours_simple(binary)

    print(f"  Total contours found (area >= 5px): {len(contours)}")

    if not contours:
        print("  No contours found! Try adjusting threshold.")
        return

    # ── Contour size statistics ──
    areas = [c['area'] for c in contours]
    bbox_ws = [c['bbox_w'] for c in contours]
    bbox_hs = [c['bbox_h'] for c in contours]

    print(f"\n--- Contour Size Distribution ---")
    print(f"  Area:   min={min(areas)}  max={max(areas)}  "
          f"mean={np.mean(areas):.0f}  median={np.median(areas):.0f}")
    print(f"  BBox W: min={min(bbox_ws)}  max={max(bbox_ws)}  "
          f"mean={np.mean(bbox_ws):.0f}  median={np.median(bbox_ws):.0f}")
    print(f"  BBox H: min={min(bbox_hs)}  max={max(bbox_hs)}  "
          f"mean={np.mean(bbox_hs):.0f}  median={np.median(bbox_hs):.0f}")

    # Area histogram
    print(f"\n--- Area Histogram ---")
    area_bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 50000, 100000]
    for i in range(len(area_bins) - 1):
        count = sum(1 for a in areas if area_bins[i] <= a < area_bins[i+1])
        if count > 0:
            bar = '#' * min(count, 50)
            print(f"  {area_bins[i]:6d}-{area_bins[i+1]:6d}: {count:4d} {bar}")
    big = sum(1 for a in areas if a >= area_bins[-1])
    if big > 0:
        print(f"  {area_bins[-1]:6d}+      : {big:4d} {'#' * min(big, 50)}")

    # ── Top 30 largest contours (likely features) ──
    sorted_contours = sorted(contours, key=lambda c: c['area'], reverse=True)
    print(f"\n--- Top 30 Largest Contours ---")
    print(f"  {'#':>3s}  {'Area':>7s}  {'BBox WxH':>10s}  {'Aspect':>6s}  "
          f"{'Centroid X':>10s}  {'Centroid Y':>10s}  {'Mean Int':>8s}")

    for i, c in enumerate(sorted_contours[:30]):
        aspect = c['bbox_w'] / max(1, c['bbox_h'])
        # Get mean intensity inside this contour's bounding box
        region = gray[c['y_min']:c['y_max']+1, c['x_min']:c['x_max']+1]
        mean_int = region.mean()

        print(f"  {i+1:3d}  {c['area']:7d}  {c['bbox_w']:4d}x{c['bbox_h']:<4d}  "
              f"{aspect:6.2f}  {c['centroid_x']:10.1f}  {c['centroid_y']:10.1f}  "
              f"{mean_int:8.1f}")

    # ── Save annotated image ──
    out_dir = os.path.dirname(path)
    base = os.path.splitext(os.path.basename(path))[0]

    # Draw bounding boxes on a copy
    annotated = rgb.copy()
    for c in sorted_contours[:50]:
        # Draw cyan rectangle (2px thick) — visible on red images
        y1, y2 = c['y_min'], c['y_max']
        x1, x2 = c['x_min'], c['x_max']
        # Top and bottom edges
        annotated[max(0,y1-1):y1+1, x1:x2+1] = [0, 255, 255]
        annotated[y2:min(h,y2+2), x1:x2+1] = [0, 255, 255]
        # Left and right edges
        annotated[y1:y2+1, max(0,x1-1):x1+1] = [0, 255, 255]
        annotated[y1:y2+1, x2:min(w,x2+2)] = [0, 255, 255]

    annotated_path = os.path.join(out_dir, f"{base}_annotated.jpg")
    if HAS_PIL:
        Image.fromarray(annotated).save(annotated_path, quality=95)
        print(f"\nAnnotated image saved: {annotated_path}")

    # ── Save stats to text file ──
    stats_path = os.path.join(out_dir, f"{base}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Image: {path}\n")
        f.write(f"Size: {w}x{h}\n")
        f.write(f"Analysis channel: {ch_name}\n")
        f.write(f"Threshold used: < {best_thresh}\n")
        f.write(f"Total contours: {len(contours)}\n")
        f.write(f"Background intensity: ~{background_level:.0f}\n\n")

        f.write(f"{'#':>3s}  {'Area':>7s}  {'BBox WxH':>10s}  {'Aspect':>6s}  "
                f"{'CentX':>8s}  {'CentY':>8s}  {'MeanInt':>7s}\n")
        f.write("-" * 65 + "\n")

        for i, c in enumerate(sorted_contours):
            aspect = c['bbox_w'] / max(1, c['bbox_h'])
            region = gray[c['y_min']:c['y_max']+1, c['x_min']:c['x_max']+1]
            mean_int = region.mean()
            f.write(f"{i+1:3d}  {c['area']:7d}  {c['bbox_w']:4d}x{c['bbox_h']:<4d}  "
                    f"{aspect:6.2f}  {c['centroid_x']:8.1f}  {c['centroid_y']:8.1f}  "
                    f"{mean_int:7.1f}\n")

    print(f"Full stats saved: {stats_path}")
    print(f"\n{'='*60}")
    print(f"DONE. Check the annotated image to see what was detected.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 feature_diagnostic.py <image_path>")
        print("  e.g.: python3 feature_diagnostic.py ~/Desktop/Users/2025.02.26/capture.jpg")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}")
        sys.exit(1)

    analyze_image(path)