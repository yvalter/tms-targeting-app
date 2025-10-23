#!/usr/bin/env python3

# ======================================================================
# [1] CONFIGURATION & CONSTANTS
# ======================================================================
import sys
import json
from pathlib import Path
import numpy as np
import trimesh
from tqdm import tqdm

# Default landmarks (used unless overridden by caller-provided FPZ/Inion/DLPFC)
LANDMARKS = {
    "Cz":   [  0.3300,   -1.9828,  94.6484],
    "Fpz":  [  0.3300, -103.3470,  -0.6190],
    "Oz":[  0.3300,  103.3470,  -0.6190],
    "dlPFC":[ 68.5660,  -52.1200,   34.4291],
}

# NOTE: Mesh path is resolved at runtime; this constant is unused but retained for numbering stability.
MESH_FILE = "mni_head.stl"

ALPHA_HEAD = [0, 0, 255, 191]   # 75% blue (visible)
EPS = 1e-9

# ======================================================================
# [2] GEOMETRY HELPERS
# ======================================================================
def ortho_frame(normal: np.ndarray):
    n = normal / (np.linalg.norm(normal) + EPS)
    u = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0.0, 1.0, 0.0])
    u /= (np.linalg.norm(u) + EPS)
    v = np.cross(n, u)
    return u, v

def plane_edge_hits(mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray, tol=1e-8):
    """
    Robust edge–plane intersections without networkx.
    Returns unique intersection points (N,3).
    """
    d = (mesh.vertices - origin) @ normal
    hits = []
    for v0, v1 in mesh.edges_unique:
        d0, d1 = d[v0], d[v1]
        if abs(d0) < tol: hits.append(mesh.vertices[v0])
        if abs(d1) < tol: hits.append(mesh.vertices[v1])
        if d0 * d1 < 0.0:
            t = d0 / (d0 - d1)  # in (0,1)
            p = mesh.vertices[v0] + t * (mesh.vertices[v1] - mesh.vertices[v0])
            hits.append(p)
    if not hits:
        return np.empty((0,3))
    uniq = np.array(list({tuple(np.round(p, 6)) for p in hits}))
    return uniq

# ======================================================================
# [3] MESH LOAD & PREP
# ======================================================================
def resolve_mesh_path() -> str:
    """
    Prefer SCALED_HEAD.stl in CWD, else fall back to mni_head.stl.
    If SCALED_HEAD.stl is missing, shout as requested.
    """
    scaled = Path("SCALED_HEAD.stl")
    if scaled.exists():
        return str(scaled)

    # Yell if scaled model is not found (even if we can fall back)
    print("SCALED_MODEL NOT FOUNDDDD!!!!!", file=sys.stderr)

    fallback = Path("mni_head.stl")
    if fallback.exists():
        return str(fallback)

    # Neither exists – raise a clear error
    raise FileNotFoundError(
        "Neither SCALED_HEAD.stl nor mni_head.stl found in the working directory."
    )

def load_and_prep_mesh(path_str: str):
    mesh = trimesh.load(path_str, process=False)
    # Outward culling & cleaning
    cent, norms = mesh.triangles_center, mesh.face_normals
    mesh.update_faces((norms * (cent - mesh.centroid)).sum(1) > 0)
    mesh.remove_unreferenced_vertices()
    mesh.visual.face_colors = ALPHA_HEAD   # make head visible (75% blue)
    return mesh

# ======================================================================
# [4] LANDMARK SNAPPING
# ======================================================================
def _as_vec3(x):
    try:
        a = np.asarray(x, dtype=float).reshape(-1)
        if a.size >= 3:
            return np.array([a[0], a[1], a[2]], dtype=float)
    except Exception:
        pass
    return None

def resolve_landmarks_with_custom(default_landmarks: dict) -> dict:
    """
    Look for caller-provided FPZ/INION/DLPFC in the __main__ module.
    If any are missing/invalid, print the required message and fall back to defaults.
    """
    caller = sys.modules.get('__main__')
    mapping = {'Fpz': 'FPZ', 'Oz': 'OZ', 'dlPFC': 'DLPFC'}

    overrides = {}
    all_found = True

    for key_lmk, key_upper in mapping.items():
        v = None
        if caller is not None and hasattr(caller, key_upper):
            v = _as_vec3(getattr(caller, key_upper))
        if v is None:
            all_found = False
            break
        else:
            overrides[key_lmk] = v

    if not all_found:
        print("CUSTOM VARIABLES NOT FOUND!!!!!!!!!!!!!!!!!!!!!!!!!", file=sys.stderr)
        return default_landmarks
    else:
        result = {
            'Cz': default_landmarks['Cz'],  # Always use default Cz
            'Fpz': overrides['Fpz'],
            'Oz': overrides['Oz'], 
            'dlPFC': overrides['dlPFC']
        }
        return result

def snap_landmarks(mesh, landmarks_dict):
    pq = trimesh.proximity.ProximityQuery(mesh)
    names = list(landmarks_dict.keys())
    raw = np.array(list(landmarks_dict.values()))
    snapped = []
    for pt in raw:
        p, _, _ = pq.on_surface([pt])
        snapped.append(p[0])
    snapped = np.vstack(snapped)
    return {n: snapped[i] for i, n in enumerate(names)}, pq

# ======================================================================
# [5] REFERENCE PLANES & inter_BC
#      A: through (Cz, Oz, Fpz)
#      H: through (Oz↔Fpz), perpendicular to A
#      C: through (dlPFC↔Cz), perpendicular to H
#      inter_BC: first scalp hit along line H∩C
# ======================================================================
def compute_planes_and_interbc(mesh, pos, pq):
    nA = np.cross(pos['Fpz'] - pos['Cz'], pos['Oz'] - pos['Cz']); nA /= (np.linalg.norm(nA) + EPS)
    line_FI = pos['Fpz'] - pos['Oz']; line_FI /= (np.linalg.norm(line_FI) + EPS)
    nH = np.cross(line_FI, nA); nH /= (np.linalg.norm(nH) + EPS)
    H_origin = pos['Fpz']

    line_DC = pos['dlPFC'] - pos['Cz']; line_DC /= (np.linalg.norm(line_DC) + EPS)
    nC = np.cross(line_DC, nH); nC /= (np.linalg.norm(nC) + EPS)

    # H ∩ C line, take first surface hit
    dir_line = np.cross(nH, nC); dir_line /= (np.linalg.norm(dir_line) + EPS)
    A_mat = np.vstack([nH, nC])
    b_vec = np.array([np.dot(nH, H_origin), np.dot(nC, pos['Cz'])])
    try:
        base = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    except np.linalg.LinAlgError:
        raise ValueError("Failed to compute plane intersection due to singular matrix.")
    
    target_vec = (pos['dlPFC'] + pos['Fpz']) * 0.5 - base
    if np.dot(dir_line, target_vec) < 0: dir_line = -dir_line
    ray_start = base + dir_line * 1.0
    locs, _, _ = mesh.ray.intersects_location([ray_start], [dir_line])
    inter_BC = locs[0] if len(locs) else pq.on_surface([base])[0][0]

    keep_sign = np.sign(np.dot(nH, pos['Cz'] - H_origin)) or 1.0  # orientation only
    return nA, nH, nC, H_origin, inter_BC, keep_sign

# ======================================================================
# [6] VERTICAL SECTOR → PATH (Cz → … → dlPFC)
#     Rays: mid→dlPFC and mid→Cz (sector not visualized; orbs only)
# ======================================================================
def build_vertical_path(mesh, pos, mid, L):
    d1 = pos['dlPFC'] - mid; d1 /= (np.linalg.norm(d1) + EPS)
    d2 = pos['Cz']    - mid; d2 /= (np.linalg.norm(d2) + EPS)
    nVert = np.cross(d1, d2); nVert /= (np.linalg.norm(nVert) + EPS)

    sect = plane_edge_hits(mesh, origin=mid, normal=nVert)
    D = np.column_stack((d1, d2))
    DTi = np.linalg.pinv(D.T @ D) @ D.T
    inside = []
    for P in sect:
        a, b = (DTi @ (P - mid)).tolist()
        if a >= -1e-6 and b >= -1e-6 and (a + b) <= (L + 1e-6):
            inside.append(P)
    inside = np.array(inside)

    # Greedy NN from Cz → … → dlPFC
    path = [pos['Cz']]
    rem = inside.copy()
    while True:
        cur = path[-1]
        if len(rem) == 0:
            path.append(pos['dlPFC']); break
        dists = np.linalg.norm(rem - cur, axis=1)
        i_min = int(np.argmin(dists))
        p_near = rem[i_min]
        if np.linalg.norm(pos['dlPFC'] - cur) <= dists[i_min]:
            path.append(pos['dlPFC']); break
        path.append(p_near)
        rem = np.delete(rem, i_min, axis=0)

    path = np.array(path)
    length = float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))
    return path, length

# ======================================================================
# [7] HORIZONTAL SECTOR → PATH (Fpz → … → inter_BC)
#     Rays: mid→oz and mid→inter_BC (sector not visualized; orbs only)
# ======================================================================
def build_horizontal_path(mesh, pos, mid, inter_BC, L):
    d1 = pos['Fpz'] - mid;  d1 /= (np.linalg.norm(d1) + EPS)
    d2 = inter_BC     - mid;  d2 /= (np.linalg.norm(d2) + EPS)
    nHoriz = np.cross(d1, d2); nHoriz /= (np.linalg.norm(nHoriz) + EPS)

    sect = plane_edge_hits(mesh, origin=mid, normal=nHoriz)
    D = np.column_stack((d1, d2))
    DTi = np.linalg.pinv(D.T @ D) @ D.T
    inside = []
    for P in sect:
        a, b = (DTi @ (P - mid)).tolist()
        if a >= -1e-6 and b >= -1e-6 and (a + b) <= (L + 1e-6):
            inside.append(P)
    inside = np.array(inside)

    # Greedy NN from Fpz → … → inter_BC
    path = [pos['Fpz']]
    rem = inside.copy()
    while True:
        cur = path[-1]
        if len(rem) == 0:
            path.append(inter_BC); break
        dists = np.linalg.norm(rem - cur, axis=1)
        i_min = int(np.argmin(dists))
        p_near = rem[i_min]
        if np.linalg.norm(inter_BC - cur) <= dists[i_min]:
            path.append(inter_BC); break
        path.append(p_near)
        rem = np.delete(rem, i_min, axis=0)

    path = np.array(path)
    length = float(np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1)))
    return path, length

# ======================================================================
# [8] PATH LENGTHS (print BEFORE render)
# ======================================================================
def print_lengths(len_vertical, len_horizontal):
    print(f"VERTICALPATH: {len_vertical:.3f}")
    print(f"HORIZONTALPATH: {len_horizontal:.3f}")

# ======================================================================
# [9] OUTPUT RESULTS (NEW - for Flask integration)
# ======================================================================
def output_results(len_vertical, len_horizontal, vertical_path, horizontal_path, output_format='text'):
    """Output results in the specified format"""
    if output_format == 'json':
        results = {
            'vertical_length': float(len_vertical),
            'horizontal_length': float(len_horizontal),
            'vertical_path': vertical_path.tolist() if isinstance(vertical_path, np.ndarray) else vertical_path,
            'horizontal_path': horizontal_path.tolist() if isinstance(horizontal_path, np.ndarray) else horizontal_path
        }
        print(json.dumps(results))
    else:
        # Original text format
        print_lengths(len_vertical, len_horizontal)

# ======================================================================
# MAIN
# ======================================================================
def main(output_format='text', silent_progress=False):
    """
    Main function with configurable output format and progress display
    
    Args:
        output_format: 'text' (original) or 'json' (for Flask integration)
        silent_progress: If True, suppress tqdm progress bars
    """
    try:
        pb_disable = silent_progress
        with tqdm(total=10, desc="Pipeline", ncols=92, disable=pb_disable) as pb:
            # [3] Mesh
            pb.set_description("[3] Resolve & prep mesh"); pb.update()
            mesh_path = resolve_mesh_path()
            mesh = load_and_prep_mesh(mesh_path)

            # Resolve landmarks with optional caller overrides (FPZ/OZ/DLPFC)
            resolved_landmarks = resolve_landmarks_with_custom(LANDMARKS)

            # [4] Landmarks
            pb.set_description("[4] Snap landmarks"); pb.update()
            pos, pq = snap_landmarks(mesh, resolved_landmarks)

            # [5] Planes & inter_BC
            pb.set_description("[5] Planes & inter_BC"); pb.update()
            nA, nH, nC, H_origin, inter_BC, keep_sign = compute_planes_and_interbc(mesh, pos, pq)

            # Common values
            base_L = 2.0 * np.linalg.norm(mesh.bounding_box.extents)
            L = 0.4 * base_L
            mid = 0.5 * (pos['Fpz'] + pos['Oz'])

            # [6] Vertical sector path
            pb.set_description("[6] Vertical path"); pb.update()
            pathV, lengthV = build_vertical_path(mesh, pos, mid, L)

            # [7] Horizontal sector path
            pb.set_description("[7] Horizontal path"); pb.update()
            pathH, lengthH = build_horizontal_path(mesh, pos, mid, inter_BC, L)

            # [8] Output results
            pb.set_description("[8] Output results"); pb.update()
            output_results(lengthV, lengthH, pathV, pathH, output_format)
            
    except Exception as e:
        # Print error to stderr so it doesn't interfere with JSON output
        print(f"ERROR: {str(e)}", file=sys.stderr)
        if output_format == 'json':
            # Output error in JSON format for Flask to parse
            error_result = {
                'error': str(e),
                'vertical_length': None,
                'horizontal_length': None,
                'vertical_path': [],
                'horizontal_path': []
            }
            print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    # Check for command line arguments
    output_format = 'text'
    silent = False
    
    if len(sys.argv) > 1:
        if '--json' in sys.argv:
            output_format = 'json'
        if '--silent' in sys.argv:
            silent = True
    
    main(output_format=output_format, silent_progress=silent)