import os
from datetime import datetime
import numpy as np
import cv2
from typing import List, Dict, Optional

# ── Configuration ──────────────────────────────────────────────────
DEFAULT_THRESHOLD = 50
MIN_FEATURE_AREA = 1500
MAX_FEATURE_AREA = 60000
CLUSTER_DISTANCE = 800
MIN_CLUSTER_SIZE = 7
MAX_GROUPS = 4
GRID_ROWS = 2
GRID_COLS = 5

# ── Image Processing (OpenCV Optimized) ───────────────────────────

def extract_red_channel(rgb: np.ndarray) -> np.ndarray:
    """Extract the red channel (index 0 in most cases for these sensors)."""
    if rgb.ndim == 2:
        return rgb
    # Use OpenCV split for speed
    channels = cv2.split(rgb)
    # Check contrast range to find the most useful channel
    best_idx = 0
    max_range = 0
    for i, ch in enumerate(channels[:3]):
        r = float(ch.max()) - float(ch.min())
        if r > max_range:
            max_range = r
            best_idx = i
    return channels[best_idx]

def threshold_image(channel: np.ndarray, thresh: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Fast binary threshold using OpenCV."""
    # Dark pixels (< thresh) become 255 (White), others become 0 (Black)
    _, binary = cv2.threshold(channel, thresh, 255, cv2.THRESH_BINARY_INV)
    return binary

def clear_border_components(stats: np.ndarray, num_labels: int, width: int, height: int) -> List[int]:
    """Returns a list of label IDs that do NOT touch the image edges."""
    valid_ids = []
    for i in range(1, num_labels):  # Skip background (0)
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        
        # If any side of the bounding box is at the pixel limit, it's touching the edge
        if (x <= 1 or y <= 1 or (x + w) >= width - 1 or (y + h) >= height - 1):
            continue
        valid_ids.append(i)
    return valid_ids

def find_components(binary: np.ndarray) -> List[Dict]:
    """Uses OpenCV's C++ labeling to find blobs in < 50ms."""
    h, w = binary.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 1. Filter out blobs touching the edge
    valid_ids = clear_border_components(stats, num_labels, w, h)
    
    components = []
    for i in valid_ids:
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 2. Area Filter
        if MIN_FEATURE_AREA < area < MAX_FEATURE_AREA:
            components.append({
                'area': area,
                'centroid_x': float(centroids[i, 0]),
                'centroid_y': float(centroids[i, 1]),
                'x_min': int(stats[i, cv2.CC_STAT_LEFT]),
                'y_min': int(stats[i, cv2.CC_STAT_TOP]),
                'x_max': int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                'y_max': int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                'bbox_w': int(stats[i, cv2.CC_STAT_WIDTH]),
                'bbox_h': int(stats[i, cv2.CC_STAT_HEIGHT]),
            })
    return components

def erode_binary(binary: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Fast OpenCV erosion."""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(binary, kernel, iterations=iterations)

# ── Clustering (Fast NumPy Distance) ───────────────────────────

def cluster_components(components: List[Dict]) -> List[List[Dict]]:
    n = len(components)
    if n == 0: return []

    # Pre-extract centroids into a numpy array for vectorized distance calc
    pts = np.array([[c['centroid_x'], c['centroid_y']] for c in components])
    assigned = -np.ones(n, dtype=int)
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0: continue
        assigned[i] = cluster_id
        queue = [i]
        
        # Breadth-first search for neighbors within CLUSTER_DISTANCE
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            # Vectorized distance from 'curr' to all other points
            dists = np.linalg.norm(pts - pts[curr], axis=1)
            neighbors = np.where((assigned == -1) & (dists <= CLUSTER_DISTANCE))[0]
            for nb in neighbors:
                assigned[nb] = cluster_id
                queue.append(nb)
        cluster_id += 1

    # Grouping
    clusters = [[] for _ in range(cluster_id)]
    for i, cid in enumerate(assigned):
        clusters[cid].append(components[i])

    valid = [c for c in clusters if len(c) >= MIN_CLUSTER_SIZE]
    valid.sort(key=lambda g: (np.mean([p['centroid_y'] for p in g]), np.mean([p['centroid_x'] for p in g])))
    return valid[:MAX_GROUPS]

# ── Grid Fitting and Decoding ─────────────────────────────────────

def fit_grid_to_cluster(cluster: List[Dict], binary: np.ndarray, 
                        eroded: np.ndarray, debug: bool = False) -> Optional[Dict]:
    if len(cluster) < MIN_CLUSTER_SIZE: return None

    # Split rows by Y
    sorted_y = sorted(cluster, key=lambda c: c['centroid_y'])
    y_vals = [c['centroid_y'] for c in sorted_y]
    best_gap_idx = np.argmax(np.diff(y_vals)) + 1
    
    row0_blobs = sorted(sorted_y[:best_gap_idx], key=lambda c: c['centroid_x'])
    row1_blobs = sorted(sorted_y[best_gap_idx:], key=lambda c: c['centroid_x'])

    # Adaptive splitting logic
    widths = sorted([b['bbox_w'] for b in cluster])
    ref_w = widths[len(widths)//4]
    bridge_thresh = ref_w * 1.4

    def process_row(blobs):
        res = []
        for b in blobs:
            if b['bbox_w'] > bridge_thresh:
                n = max(2, round(b['bbox_w'] / (ref_w * 1.2)))
                pw = (b['x_max'] - b['x_min']) / n
                for p in range(n):
                    x_start = int(b['x_min'] + p * pw)
                    x_end = int(b['x_min'] + (p+1) * pw)
                    # Crop and measure height on eroded image
                    crop = eroded[b['y_min']:b['y_max'], x_start:x_end]
                    ys, _ = np.where(crop == 255)
                    if len(ys) > 0:
                        res.append({
                            'centroid_x': (x_start + x_end) / 2.0,
                            'centroid_y': b['y_min'] + ys.mean(),
                            'bbox_w': x_end - x_start,
                            'bbox_h': ys.max() - ys.min(),
                            'area': len(ys),
                            'y_min': b['y_min'] + ys.min(), 'y_max': b['y_min'] + ys.max(),
                            'x_min': x_start, 'x_max': x_end
                        })
            else:
                # Measure on eroded image for shape classification
                crop = eroded[b['y_min']:b['y_max'], b['x_min']:b['x_max']]
                ys, _ = np.where(crop == 255)
                if len(ys) > 0:
                    b['area'] = len(ys)
                    b['bbox_h'] = ys.max() - ys.min()
                    res.append(b)
        return sorted(res, key=lambda x: x['centroid_x'])[:GRID_COLS]

    r0 = process_row(row0_blobs)
    r1 = process_row(row1_blobs)
    if len(r0) < GRID_COLS or len(r1) < GRID_COLS: return None

    # Adaptive classification: Shape = Lower Aspect + Higher Area
    all_data = []
    for r_idx, row in enumerate([r0, r1]):
        for c_idx, b in enumerate(row):
            aspect = b['bbox_w'] / max(1, b['bbox_h'])
            all_data.append((r_idx, c_idx, aspect, b['area']))

    asp_min, asp_max = min(d[2] for d in all_data), max(d[2] for d in all_data)
    are_min, are_max = min(d[3] for d in all_data), max(d[3] for d in all_data)

    asp_range = asp_max - asp_min
    are_range = are_max - are_min
    avg_area = np.mean([d[3] for d in all_data])
    avg_aspect = np.mean([d[2] for d in all_data])

    # Uniform shapes = all same type (0,0 or 31,31)
    # Mixed markers have large spread in both aspect and area
    # Threshold: mixed markers typically have area range > 5000 and aspect range > 0.2
    is_uniform = (are_range < 4000) and (asp_range < 0.25)

    if is_uniform:
        if avg_area < 10000:
            col_id, row_id = 0, 0
        else:
            col_id, row_id = 31, 31
        confidence = 0.0
        if debug:
            print(f"  Uniform shapes: asp_range={asp_range:.3f} are_range={are_range:.0f} "
                  f"avg_area={avg_area:.0f} avg_asp={avg_aspect:.2f} -> ({col_id},{row_id})")
    else:
        scores = {}
        for r, c, asp, area in all_data:
            n_asp = (asp - asp_min) / max(0.001, (asp_max - asp_min))
            n_are = 1.0 - (area - are_min) / max(0.001, (are_max - are_min))
            scores[(r, c)] = n_asp + n_are

        sorted_sc = sorted(scores.values())
        split_idx = np.argmax(np.diff(sorted_sc))
        confidence = float(sorted_sc[split_idx+1] - sorted_sc[split_idx])

        thresh = (sorted_sc[split_idx] + sorted_sc[split_idx+1]) / 2.0
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        for (r, c), s in scores.items():
            grid[r, c] = 1 if s < thresh else 0

        col_id, row_id = 0, 0
        for c in range(GRID_COLS):
            col_id |= (grid[0, c] << c)
            row_id |= (grid[1, c] << c)

        if debug:
            print(f"  Mixed: asp_range={asp_range:.3f} are_range={are_range:.0f} "
                  f"confidence={confidence:.3f} -> ({col_id},{row_id})")

    return {
        'center': (np.mean([b['centroid_x'] for b in cluster]), np.mean([b['centroid_y'] for b in cluster])),
        'col_id': col_id, 'row_id': row_id,
        'confidence': confidence,
        'bbox': (min(b['x_min'] for b in cluster), min(b['y_min'] for b in cluster),
                 max(b['x_max'] for b in cluster), max(b['y_max'] for b in cluster))
    }

def detect_markers(rgb: np.ndarray, threshold: int = DEFAULT_THRESHOLD, debug: bool = False) -> List[Dict]:
    channel = extract_red_channel(rgb)
    binary = threshold_image(channel, threshold)
    components = find_components(binary)
    if not components: return []
    
    clusters = cluster_components(components)
    eroded = erode_binary(binary)
    
    markers = []
    for c in clusters:
        m = fit_grid_to_cluster(c, binary, eroded, debug)
        if m: markers.append(m)
    return markers

FAILED_IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed_images")

def save_failed_frame(frame: np.ndarray, tag: str = "") -> Optional[str]:
    """Save a frame that failed marker detection to ./failed_images/ for offline diagnosis."""
    if frame is None:
        return None
    try:
        os.makedirs(FAILED_IMAGES_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(FAILED_IMAGES_DIR, f"fail_{ts}{suffix}.jpg")
        from PIL import Image
        Image.fromarray(frame).save(path, quality=90)
        return path
    except Exception as e:
        print(f"[save_failed_frame] error: {e}")
        return None


if __name__ == '__main__':
    import sys
    from PIL import Image

    if len(sys.argv) < 2:
        print("Usage: python feature_detection.py <image_path> [threshold]")
        sys.exit(1)

    img_path = sys.argv[1]
    thresh = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD

    rgb = np.array(Image.open(img_path).convert('RGB'))
    h, w = rgb.shape[:2]
    print(f"Image: {img_path} ({w}x{h})")
    print(f"Threshold: {thresh}")

    channel = extract_red_channel(rgb)
    print(f"Best channel range: {float(channel.max()) - float(channel.min()):.0f}")

    binary = threshold_image(channel, thresh)
    white_pct = np.sum(binary == 255) / binary.size * 100
    print(f"Binary: {white_pct:.1f}% white pixels")

    components = find_components(binary)
    print(f"\nComponents found: {len(components)}")
    for i, c in enumerate(components):
        print(f"  [{i}] area={c['area']} px=({c['centroid_x']:.0f},{c['centroid_y']:.0f}) "
              f"bbox={c['bbox_w']}x{c['bbox_h']}")

    clusters = cluster_components(components)
    print(f"\nClusters: {len(clusters)}")
    for i, cl in enumerate(clusters):
        xs = [b['centroid_x'] for b in cl]
        ys = [b['centroid_y'] for b in cl]
        print(f"  Cluster {i}: {len(cl)} blobs, center=({np.mean(xs):.0f},{np.mean(ys):.0f})")

    eroded = erode_binary(binary)
    markers = []
    for i, cl in enumerate(clusters):
        m = fit_grid_to_cluster(cl, binary, eroded, debug=True)
        if m:
            markers.append(m)
            print(f"\n  Marker {i}: col_id={m['col_id']} row_id={m['row_id']} "
                  f"confidence={m['confidence']:.3f} "
                  f"center=({m['center'][0]:.0f},{m['center'][1]:.0f})")
        else:
            print(f"\n  Cluster {i}: FAILED to decode (not enough blobs per row?)")

    print(f"\n=== SUMMARY: {len(markers)} markers detected ===")
    for m in markers:
        print(f"  ({m['col_id']},{m['row_id']}) at px ({m['center'][0]:.0f},{m['center'][1]:.0f}) "
              f"conf={m['confidence']:.3f}")

    # Save output
    out_path = img_path.rsplit('.', 1)[0] + '_detection.txt'
    with open(out_path, 'w') as f:
        f.write(f"Image: {img_path} ({w}x{h})\n")
        f.write(f"Threshold: {thresh}\n")
        f.write(f"Components: {len(components)}\n")
        f.write(f"Clusters: {len(clusters)}\n")
        f.write(f"Markers: {len(markers)}\n\n")
        for m in markers:
            f.write(f"({m['col_id']},{m['row_id']}) px=({m['center'][0]:.0f},{m['center'][1]:.0f}) "
                    f"conf={m['confidence']:.3f}\n")
    print(f"\nSaved to {out_path}")