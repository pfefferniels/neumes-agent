import numpy as np
import math
from collections import deque
from PIL import Image, ImageDraw, ImageOps, ImageFilter

# ------------------------
# 1. BINARIZATION
# ------------------------

def load_and_binarize(path,
                      do_blur=True,
                      do_closing=False,
                      do_opening=True):
    """
    Improved binarization for parchment-like backgrounds.

    Returns a uint8 array with values {0,1}, where 1 = foreground (ink).
    """

    # 1) Load & desaturate
    img = Image.open(path).convert("L")  # grayscale

    # 2) Enhance contrast / exposure
    img = ImageOps.autocontrast(img, cutoff=2)

    arr = np.array(img).astype(np.uint8)

    # 3) Otsu thresholding
    hist, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)

    sumB = 0.0
    wB = 0.0
    max_between = 0.0
    threshold = 0

    for t in range(256):
        wB += hist[t]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum_total - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > max_between:
            max_between = between
            threshold = t
    print("Otsu threshold:", threshold)
    threshold = threshold  # bias towards ink

    # Optional: small blur to suppress grainy background noise
    if do_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.7))

    # Ink is darker
    binary = (arr < threshold).astype(np.uint8)

    # 4a) Optional: small closing to *connect* thin parts
    if do_closing:
        try:
            from scipy.ndimage import binary_closing
            binary = binary_closing(binary, structure=np.ones((3, 3))).astype(np.uint8)
        except ImportError:
            pass  # if scipy isn't available, just skip

    # 4b) Optional: opening for speckle removal (OFF by default!)
    if do_opening:
        try:
            from scipy.ndimage import binary_opening
            binary = binary_opening(binary, structure=np.ones((2, 2))).astype(np.uint8)
        except ImportError:
            pass

    return binary


# ------------------------
# 2. ZHANG–SUEN SKELETONIZATION
# ------------------------

def neighbours(x, y, image):
    """Return 8-neighbours of image point P1(x,y), in order P2..P9."""
    img = image
    return [
        img[x-1, y],   # P2
        img[x-1, y+1], # P3
        img[x,   y+1], # P4
        img[x+1, y+1], # P5
        img[x+1, y],   # P6
        img[x+1, y-1], # P7
        img[x,   y-1], # P8
        img[x-1, y-1], # P9
    ]


def transitions(neigh):
    """Number of 0->1 transitions in circular sequence of neighbours."""
    n = neigh + [neigh[0]]
    return sum((n1 == 0 and n2 == 1) for n1, n2 in zip(n[:-1], n[1:]))


def zhang_suen_thinning(image):
    """
    Skeletonize a binary image using the Zhang–Suen thinning algorithm.
    Input: uint8 array with values {0,1}. 1 = foreground.
    Output: same shape & type.
    """
    img = image.copy()
    changing1 = changing2 = True

    while changing1 or changing2:
        rows, cols = img.shape
        changing1 = []
        # Step 1
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                P1 = img[x, y]
                if P1 != 1:
                    continue
                neigh = neighbours(x, y, img)
                C = transitions(neigh)
                N = sum(neigh)
                if (
                    2 <= N <= 6 and
                    C == 1 and
                    neigh[0] * neigh[2] * neigh[4] == 0 and
                    neigh[2] * neigh[4] * neigh[6] == 0
                ):
                    changing1.append((x, y))
        for x, y in changing1:
            img[x, y] = 0

        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                P1 = img[x, y]
                if P1 != 1:
                    continue
                neigh = neighbours(x, y, img)
                C = transitions(neigh)
                N = sum(neigh)
                if (
                    2 <= N <= 6 and
                    C == 1 and
                    neigh[0] * neigh[2] * neigh[6] == 0 and
                    neigh[0] * neigh[4] * neigh[6] == 0
                ):
                    changing2.append((x, y))
        for x, y in changing2:
            img[x, y] = 0

    return img

def bresenham_line(r0, c0, r1, c1):
    """Yield (r,c) pixels on a Bresenham line."""
    # classic integer Bresenham
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1

    if dc > dr:
        err = dc // 2
        while c0 != c1:
            yield r0, c0
            err -= dr
            if err < 0:
                r0 += sr
                err += dc
            c0 += sc
        yield r0, c0
    else:
        err = dr // 2
        while r0 != r1:
            yield r0, c0
            err -= dc
            if err < 0:
                c0 += sc
                err += dr
            r0 += sr
        yield r0, c0


def close_skeleton_gaps(skel, max_gap=10.0, angle_thresh_deg=30.0):
    """
    Attempt to connect broken strokes on a skeleton image by joining
    endpoints that are:
      - within max_gap pixels,
      - in different components,
      - directionally consistent (angle between local stroke direction
        and join direction < angle_thresh_deg).

    skel: uint8 array with {0,1}, 1 = skeleton foreground.
    Returns a *new* skeleton array (copy).
    """
    skel = skel.copy()

    # Build graph
    coords, adj = skeleton_to_graph(skel)
    if len(coords) == 0:
        return skel

    # Connected components in the graph
    components = connected_components(adj)
    # map node -> component id
    node_to_comp = {}
    for cid, comp in enumerate(components):
        for idx in comp:
            node_to_comp[idx] = cid

    # Degrees & endpoints
    degrees = [len(nbs) for nbs in adj]
    endpoints = [i for i, d in enumerate(degrees) if d == 1]

    if len(endpoints) < 2:
        return skel  # nothing to bridge

    # Helper to get local direction at an endpoint:
    # vector from its single neighbor to itself
    def endpoint_direction(idx):
        nbrs = adj[idx]
        if not nbrs:
            return None
        # since it's an endpoint, it should have exactly 1 neighbor
        nb = nbrs[0]
        r0, c0 = coords[nb]
        r1, c1 = coords[idx]
        v = np.array([float(r1 - r0), float(c1 - c0)])
        n = np.linalg.norm(v)
        if n == 0:
            return None
        return v / n

    def angle_between(v1, v2):
        dot = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 180.0
        cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
        return math.degrees(math.acos(cos_a))

    angle_thresh = angle_thresh_deg

    # Precompute endpoint directions
    endpoint_dirs = {i: endpoint_direction(i) for i in endpoints}

    # Try bridging endpoint pairs
    H, W = skel.shape
    for i_idx in range(len(endpoints)):
        i = endpoints[i_idx]
        ci = node_to_comp.get(i, -1)
        ri, ci_pix = coords[i]

        for j_idx in range(i_idx + 1, len(endpoints)):
            j = endpoints[j_idx]
            cj = node_to_comp.get(j, -1)
            rj, cj_pix = coords[j]

            # Only bridge between *different* components
            if ci == cj:
                continue

            # Distance check
            dr = float(ri - rj)
            dc = float(ci_pix - cj_pix)
            dist = math.hypot(dr, dc)
            if dist > max_gap:
                continue

            # Join direction (from i to j)
            join_vec = np.array([float(rj - ri), float(cj_pix - ci_pix)])
            join_dir = join_vec / (np.linalg.norm(join_vec) + 1e-9)

            di = endpoint_dirs.get(i, None)
            dj = endpoint_dirs.get(j, None)
            if di is None or dj is None:
                continue

            # Direction consistency: endpoint direction *and* join direction
            # should be roughly aligned (collinear, maybe opposite is okay).
            ai = angle_between(di, join_dir)
            aj = angle_between(dj, -join_dir)  # other endpoint faces back along the join

            if ai > angle_thresh or aj > angle_thresh:
                continue

            # Passed all tests -> draw a line between the endpoints on the skeleton
            for r, c in bresenham_line(ri, ci_pix, rj, cj_pix):
                if 0 <= r < H and 0 <= c < W:
                    skel[r, c] = 1

    return skel


# ------------------------
# 3. DROP BRANCHES (KEEP MAIN STROKE)
# ------------------------

def skeleton_to_graph(skel):
    """
    Build a graph from a skeleton image (1 = foreground).
    Returns:
      coords: list of (row, col) for each node
      adj: adjacency list (list of lists of neighbour indices)
    """
    coords = np.argwhere(skel == 1)  # (row, col)
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}

    adj = [[] for _ in range(len(coords))]
    dirs = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    for i, (r, c) in enumerate(coords):
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            j = coord_to_idx.get((nr, nc))
            if j is not None:
                adj[i].append(j)

    return coords, adj

def component_touches_border(comp_nodes, coords, h, w, margin=0):
    """
    Return True if any pixel in this component lies on the image border
    (within 'margin' pixels).
    coords: array of (row, col)
    h, w: image height & width
    margin: 0 means exactly at edge, 1 means last/first 2 pixel rows/cols, etc.
    """
    top = margin
    left = margin
    bottom = h - 1 - margin
    right = w - 1 - margin

    for idx in comp_nodes:
        r, c = coords[idx]
        if r <= top or r >= bottom or c <= left or c >= right:
            return True
    return False

def connected_components(adj):
    """
    Find connected components in the skeleton graph.
    adj: adjacency list (list of lists of neighbour indices)
    Returns a list of components, each a list of node indices.
    """
    n = len(adj)
    visited = [False] * n
    components = []

    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        components.append(comp)

    return components

def shortest_path(start, goal, adj, allowed=None):
    """
    Unweighted shortest path by BFS; returns list of node indices.
    'allowed' is an optional set of node indices we are allowed to visit.
    """
    if allowed is None:
        allowed = set(range(len(adj)))

    q = deque([start])
    prev = {start: None}
    while q:
        v = q.popleft()
        if v == goal:
            break
        for nb in adj[v]:
            if nb not in allowed:
                continue
            if nb not in prev:
                prev[nb] = v
                q.append(nb)

    if goal not in prev:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    return list(reversed(path))

def longest_endpoint_path(coords, adj, nodes_subset=None):
    """
    Find the longest shortest path between any two endpoints (degree 1)
    in the given subset of nodes. Used as the main stroke for that component.

    nodes_subset: iterable of node indices belonging to this component.
                  If None, uses all nodes.
    Returns list of node indices in order.
    """
    if nodes_subset is None:
        nodes_subset = range(len(coords))
    subset = set(nodes_subset)

    # degrees computed globally, then restricted to subset
    degrees = [len(nbs) for nbs in adj]
    endpoints = [i for i in subset if degrees[i] == 1]

    # If there are fewer than 2 endpoints (e.g. loop or dot),
    # just do a simple walk through the subset to get an ordered path.
    if len(endpoints) < 2:
        if not subset:
            return []
        start = min(subset)  # arbitrary but stable
        path = [start]
        visited = {start}
        current = start
        prev = None
        while True:
            neighbors = [nb for nb in adj[current] if nb in subset and nb != prev]
            next_nodes = [nb for nb in neighbors if nb not in visited]
            if not next_nodes:
                break
            nxt = next_nodes[0]
            path.append(nxt)
            visited.add(nxt)
            prev, current = current, nxt
        return path

    best_path = []
    best_len = -1
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            s, g = endpoints[i], endpoints[j]
            # restrict BFS to this component
            path = shortest_path(s, g, adj, allowed=subset)
            if path is None:
                continue
            if len(path) > best_len:
                best_len = len(path)
                best_path = path

    return best_path



# ------------------------
# 4. SMOOTH / SIMPLIFY POLYLINE (RDP)
# ------------------------

def rdp(points, epsilon):
    """
    Ramer–Douglas–Peucker simplification.
    points: list of (x, y)
    epsilon: distance tolerance
    """
    if len(points) < 3:
        return points

    (x1, y1) = points[0]
    (x2, y2) = points[-1]

    def perp_dist(p):
        x0, y0 = p
        num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        den = np.hypot(y2 - y1, x2 - x1)
        return num / den if den != 0 else 0.0

    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i])
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


# ------------------------
# 5. VERTICAL MOVEMENT SEQUENCE
# ------------------------

def path_length(points):
    """Euclidean length of the polyline."""
    length = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

def choose_num_segments(points,
                        pixels_per_segment=30.0,
                        min_segments=4,
                        max_segments=20):
    """
    Decide how many movement segments to use based on the total stroke length.
    Roughly one segment per 'pixels_per_segment' of arc length.
    """
    L = path_length(points)
    # at least min_segments, at most max_segments
    n = max(min_segments, min(max_segments, int(round(L / pixels_per_segment))))
    # ensure at least 2 so we can say *something*
    return max(2, n)

def resample_by_length(points, num_samples):
    import numpy as np
    import math

    if len(points) < 2:
        return points

    d = [0.0]
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        d.append(d[-1] + math.hypot(x2 - x1, y2 - y1))
    total = d[-1]
    if total == 0:
        return points

    samples = []
    target_dists = np.linspace(0, total, num_samples)
    j = 0
    for td in target_dists:
        while j < len(d) - 2 and d[j+1] < td:
            j += 1
        t = (td - d[j]) / (d[j+1] - d[j] + 1e-9)
        x = points[j][0] + t * (points[j+1][0] - points[j][0])
        y = points[j][1] + t * (points[j+1][1] - points[j][1])
        samples.append((x, y))
    return samples


def vertical_movement_sequence_auto(points,
                                    pixels_per_segment=10.0,
                                    straight_tol=3.0,
                                    min_segments=4,
                                    max_segments=20):
    # ensure left → right
    if points[0][0] > points[-1][0]:
        points = list(reversed(points))

    num_segments = choose_num_segments(points,
                                       pixels_per_segment=pixels_per_segment,
                                       min_segments=min_segments,
                                       max_segments=max_segments)

    # We need N+1 samples for N segments
    samples = resample_by_length(points, num_segments + 1)

    moves = []
    for i in range(num_segments):
        y1 = samples[i][1]
        y2 = samples[i+1][1]
        dy = y2 - y1  # image coords: down is positive

        if abs(dy) <= straight_tol:
            moves.append("s")
        elif dy < 0:
            moves.append("u")
        else:
            moves.append("d")
    return moves

def moves_to_runs(moves):
    """
    Convert ['d','d','u','u','u'] -> [('d',2), ('u',3)]
    """
    if not moves:
        return []

    runs = []
    current_dir = moves[0]
    length = 1

    for m in moves[1:]:
        if m == current_dir:
            length += 1
        else:
            runs.append((current_dir, length))
            current_dir = m
            length = 1
    runs.append((current_dir, length))
    return runs

def normalize_run_lengths(runs):
    """
    [('d',2), ('u',5), ('d',2)] -> [('d', 0.2), ('u', 0.5), ('d', 0.2)]
    """
    total = sum(L for _, L in runs)
    if total == 0:
        return [(d, 0.0) for (d, _) in runs]
    return [(d, L / total) for (d, L) in runs]

def bucket_length(frac):
    """
    Map a normalized length fraction to a bucket: 'S', 'M', or 'L'.
    """
    if frac < 0.25:
        return "S"  # short
    elif frac < 0.5:
        return "M"  # medium
    else:
        return "L"  # long

def runs_to_tokens(runs):
    """
    [('d',2), ('u',5), ('d',2)] 
        -> after normalization maybe [('d',0.22), ('u',0.55), ('d',0.22)]
        -> ['dS', 'uL', 'dS']
    """
    norm = normalize_run_lengths(runs)
    tokens = []
    for d, frac in norm:
        if d == 's':
            # you might want to ignore 's' here or make it its own symbol
            continue
        bucket = bucket_length(frac)
        tokens.append(d + bucket)  # 'uL', 'dS', ...
    return tokens

def stroke_x_position(stroke):
    xs = [p[0] for p in stroke["points"]]
    return min(xs) if xs else float("inf")

def stroke_to_tokens(stroke):
    if stroke["type"] == "dot":
        return ["0"]  # single dot symbol
    runs = moves_to_runs(stroke["moves"])
    return runs_to_tokens(runs)  # list like ['uL','dS']

def strokes_to_symbol_tokens(strokes):
    strokes = sorted(strokes, key=stroke_x_position)
    tokens = []
    for s in strokes:
        tokens.extend(stroke_to_tokens(s))
    return tokens  # e.g. ['uL','dS','0','0']

# ------------------------
# MAIN PIPELINE
# ------------------------

def process_image(
    path,
    rdp_epsilon=2.0,
    save_debug_prefix=None,
):
    """
    Full pipeline:
      1. binarize
      2. trace skeleton
      3. split into connected components (strokes)
      4. for each stroke:
           - keep its main path (drop branches)
           - simplify polyline (RDP)
           - compute vertical movement sequence

    Returns:
      strokes: list of dicts, each:
          {
            "points": simplified_points (list of (x,y)),
            "moves":  movement_sequence (list of 'u','d','s'),
          }
    """

    # 1) Binarize
    binary = load_and_binarize(path)

    # 2) Skeletonize
    skel = zhang_suen_thinning(binary)

    skel = close_skeleton_gaps(skel, max_gap=50.0, angle_thresh_deg=40.0)

    # 3) Graph + connected components
    coords, adj = skeleton_to_graph(skel)

    components = connected_components(adj)

    strokes = []

    for comp_id, comp_nodes in enumerate(components):
        if len(comp_nodes) == 0:
            continue

        # --- skip components coming from outside ----
        h, w = skel.shape
        if component_touches_border(comp_nodes, coords, h, w, margin=0):
            # we only want to skip if this is *small* (likely an unrelated dot)
            # so we need a quick length estimate for this component

            # quick path: approximate length as sqrt of bounding box diag
            rs = [coords[i][0] for i in comp_nodes]
            cs = [coords[i][1] for i in comp_nodes]
            bbox_h = max(rs) - min(rs) + 1
            bbox_w = max(cs) - min(cs) + 1
            approx_len = (bbox_h**2 + bbox_w**2) ** 0.5

            EDGE_IGNORE_THRESHOLD = 100.0  # tune; in pixels
            if approx_len < EDGE_IGNORE_THRESHOLD:
                # likely an unrelated speck/dot from a neighbouring symbol
                continue

        # 3b) main path for this component (branch removal)
        main_indices = longest_endpoint_path(coords, adj, nodes_subset=comp_nodes)
        if not main_indices:
            continue

        # coords are (row, col) = (y, x)
        main_points = [(int(coords[i][1]), int(coords[i][0])) for i in main_indices]

        # IGNORE_THRESHOLD = 1.0  # pixels
        # if path_length(main_points) < IGNORE_THRESHOLD:
        #     continue

        DOT_THRESHOLD = 40.0  # pixels of skeleton length
        stroke_len = path_length(main_points)

        if stroke_len < DOT_THRESHOLD:
            strokes.append({
                "type": "dot",
                "points": main_points,
                "moves": [],
            })
            continue

        # 4) Simplify/smooth this stroke
        simplified_points = rdp(main_points, rdp_epsilon)

        # 5) Movement sequence for this stroke
        movement_sequence = vertical_movement_sequence_auto(simplified_points)

        strokes.append({
            "type": "stroke",
            "points": simplified_points,
            "moves": movement_sequence,
        })
    
    # Optional: debug images (all strokes together)
    if save_debug_prefix is not None:
        # binary image
        bin_img = Image.fromarray((1 - binary) * 255).convert("L")
        bin_img.save(f"{save_debug_prefix}_binary.png")

        # skeleton image
        skel_img = Image.fromarray((1 - skel) * 255).convert("L")
        skel_img.save(f"{save_debug_prefix}_skeleton.png")

        # all main strokes
        h, w = skel.shape
        main_img = Image.new("L", (w, h), 255)
        draw_main = ImageDraw.Draw(main_img)
        for stroke in strokes:
            pts = stroke["points"]
            if len(pts) > 1:
                draw_main.line(pts, fill=0, width=1)
        main_img.save(f"{save_debug_prefix}_main_strokes.png")

    return strokes

LEXICON = {
    "punctum":          ["0"],
    "pes":              ["dS", "uL"],
    "pes subbipunctis": ["dS", "uL", "0", "0"],
    "clivis":           ["uM", "dM"],
    "torculus":         ["dS", "uM", "dS"],
    "virga":            ["uL"],
    "bivirga":          ['uL', 'uL'],
    "trivirga":         ['uL', 'uL', 'uL'],
    "pressus":          ["uL", "dS", "uS", "0"],
    "climacus":         ["uL", "0", "0"],
    "porrectus":        ['uS', 'dS', 'uL'],
    "scandicus":        ['0', '0', '0', 'uL'],
    "scandicus flexus": ['0', '0', 'uL', 'dL'],
    "stropha":          ['uS', 'dL'],
    "bistropha":        ['uM', 'dL', 'uM', 'dL'],
    "tristropha":       ['uM', 'dL', 'uM', 'dL', 'uM', 'dL'],
    "torculus resupinus": ['dS', 'uM', 'dS', 'uL'],
    "porrectus flexus": ['uS', 'dS', 'uM', 'dM'],
    "uncinus":          ['uL', 'dS', 'uS'],
    "bistropha(2)":     ['0', '0'],
    "tristropha(2)":    ['0', '0', '0'],
    "celeriter":        ['dL', 'uS'],
}

def token_match_score(a, b):
    """
    a, b are tokens like 'uL', 'dS', 'sM' or '0'.
    Direction: u / d / s
    Magnitude: S / M / L
    """

    # Dots: keep your existing behaviour
    if a == "0" or b == "0":
        return 2 if a == b else -1

    dir_a, mag_a = a[0], a[1]
    dir_b, mag_b = b[0], b[1]

    # Same direction (including both 's')
    if dir_a == dir_b:
        if mag_a == mag_b:
            return 3    # perfect match: same direction & magnitude
        else:
            return 1    # same direction, different magnitude

    # Different directions from here on

    # If one of them is 's' (straight) and the other is 'u' or 'd',
    # treat this as a "soft" mismatch.
    if "s" in (dir_a, dir_b):
        return -1       # mild penalty: straight vs curved

    # True opposite movement: 'u' vs 'd'
    return -3           # strong penalty


def needleman_wunsch_tokens(seq1, seq2, gap=-2):
    """
    Global alignment on token lists (not characters).
    """
    n = len(seq1)
    m = len(seq2)
    dp = [[0]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap

    for i in range(1, n+1):
        for j in range(1, m+1):
            score_diag = dp[i-1][j-1] + token_match_score(seq1[i-1], seq2[j-1])
            score_up   = dp[i-1][j] + gap
            score_left = dp[i][j-1] + gap
            dp[i][j] = max(score_diag, score_up, score_left)

    return dp[n][m]

def classify_neume_tokens(tokens, lexicon=LEXICON, n=5):
    scores = {}

    for name, pattern in lexicon.items():
        s = needleman_wunsch_tokens(tokens, pattern)
        scores[name] = s

    # Sort by score descending
    sorted_matches = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top n matches
    top_n = sorted_matches[:n]
    
    best_name = top_n[0][0] if top_n else None
    best_score = top_n[0][1] if top_n else float("-inf")

    return best_name, best_score, top_n

#if __name__ == "__main__":
#    strokes = process_image(
#        "image.png",
#        rdp_epsilon=2.0,
#        save_debug_prefix="debug",
#    )
#
#    tokens  = strokes_to_symbol_tokens(strokes)
#    print("Tokens:", tokens)
#
#    name, score, scores = classify_neume_tokens(tokens)
#    print("Best match:", name, "score:", score)

# ------------------------
# MINI WEBSERVER
# ------------------------

from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# Change this to the path of the page image you want to test on
TEST_IMAGE_PATH = "image.png"

@app.route("/")
def index():
    return send_file("index.html")

@app.route("/image")
def image():
    return send_file(TEST_IMAGE_PATH)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    x0 = int(data.get("x0", 0))
    y0 = int(data.get("y0", 0))
    x1 = int(data.get("x1", 0))
    y1 = int(data.get("y1", 0))

    # Crop the selected region
    im = Image.open(TEST_IMAGE_PATH)
    # clamp coords
    x0 = max(0, min(x0, im.width-1))
    x1 = max(0, min(x1, im.width))
    y0 = max(0, min(y0, im.height-1))
    y1 = max(0, min(y1, im.height))

    if x1 <= x0 or y1 <= y0:
        return jsonify({"error": "Invalid selection"}), 400

    crop = im.crop((x0, y0, x1, y1))
    crop_path = "_tmp_crop.png"
    crop.save(crop_path)

    # Run your pipeline
    strokes = process_image(
        crop_path,
        rdp_epsilon=2.0,
        save_debug_prefix="debug",  # or "debug_crop" if you want files
    )

    # Build tokens & classify
    tokens = strokes_to_symbol_tokens(strokes)
    best_name, best_score, top_n = classify_neume_tokens(tokens)

    return jsonify({
        "bbox": [x0, y0, x1, y1],
        "tokens": tokens,
        "best": best_name,
        "best_score": best_score,
        "strokes": [
            {
                "type": s["type"],
                "moves": s["moves"],
                "num_points": len(s["points"]),
            }
            for s in strokes
        ],
        "top_n": top_n
    })

if __name__ == "__main__":
    # Run the small dev server
    app.run(debug=True)
