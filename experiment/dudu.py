import numpy as np
import math
from collections import deque
from PIL import Image, ImageDraw, ImageOps, ImageFilter

# ------------------------
# 1. BINARIZATION
# ------------------------

def load_and_binarize(path, speckle_min_area=5):
    """
    Improved binarization for parchment-like backgrounds.

    Returns a uint8 array with values {0,1}, where 1 = foreground (ink).
    """

    # 1) Load & desaturate
    img = Image.open(path).convert("L")  # grayscale (like saturation = 0)

    # 2) Enhance contrast / exposure
    #    autocontrast with a small cutoff trims extreme shadows/highlights
    #    which is similar in spirit to your Preview tweaks.
    img = ImageOps.autocontrast(img, cutoff=2)

    # Optional: small blur to suppress grainy background noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    arr = np.array(img).astype(np.uint8)

    # 3) Otsu thresholding (global, but much smarter than mean)
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

    # Ink is darker, so we keep pixels < threshold
    binary = (arr < threshold).astype(np.uint8)

    # 4) Very simple speckle cleanup: remove tiny isolated dots
    if speckle_min_area > 1:
        # naive 3x3 “opening”: erode then dilate
        from scipy.ndimage import binary_opening  # if available
        binary = binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)

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


def shortest_path(start, goal, adj):
    """Unweighted shortest path by BFS; returns list of node indices."""
    q = deque([start])
    prev = {start: None}
    while q:
        v = q.popleft()
        if v == goal:
            break
        for nb in adj[v]:
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


def longest_endpoint_path(coords, adj):
    """
    Find the longest shortest path between any two endpoints (degree 1)
    of the skeleton graph. This is used as the main stroke.
    Returns list of node indices in order.
    """
    degrees = [len(nbs) for nbs in adj]
    endpoints = [i for i, d in enumerate(degrees) if d == 1]

    if len(endpoints) < 2:
        # Nothing to do; return all nodes as a single path
        return list(range(len(coords)))

    best_path = []
    best_len = -1
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            s, g = endpoints[i], endpoints[j]
            path = shortest_path(s, g, adj)
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
                                    pixels_per_segment=30.0,
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
      3. keep main stroke only
      4. simplify polyline
      5. compute vertical movement sequence

    Returns:
      simplified_points: list of (x, y)
      movement_sequence: list of 'u', 'd', 's'
    """

    # 1) Binarize
    binary = load_and_binarize(path)

    # 2) Skeletonize
    skel = zhang_suen_thinning(binary)

    # 3) Drop branches: build graph & keep longest endpoint path
    coords, adj = skeleton_to_graph(skel)
    main_indices = longest_endpoint_path(coords, adj)

    # Build ordered polyline (x,y) from main_indices
    # coords are (row, col) = (y, x)
    main_points = [(int(coords[i][1]), int(coords[i][0])) for i in main_indices]

    # 4) Simplify/smooth polyline with RDP
    simplified_points = rdp(main_points, rdp_epsilon)

    # 5) Movement sequence
    movement_sequence = vertical_movement_sequence_auto(
        simplified_points
    )

    # Optional: save debug images
    if save_debug_prefix is not None:
        # binary image
        bin_img = Image.fromarray((1 - binary) * 255).convert("L")
        bin_img.save(f"{save_debug_prefix}_binary.png")

        # skeleton image (black on white)
        skel_img = Image.fromarray((1 - skel) * 255).convert("L")
        skel_img.save(f"{save_debug_prefix}_skeleton.png")

        # main stroke
        h, w = skel.shape
        main_img = Image.new("L", (w, h), 255)
        draw_main = ImageDraw.Draw(main_img)
        if len(main_points) > 1:
            draw_main.line(main_points, fill=0, width=1)
        main_img.save(f"{save_debug_prefix}_main_stroke.png")

        # simplified stroke
        simple_img = Image.new("L", (w, h), 255)
        draw_simple = ImageDraw.Draw(simple_img)
        if len(simplified_points) > 1:
            draw_simple.line(simplified_points, fill=0, width=1)
        simple_img.save(f"{save_debug_prefix}_simplified.png")

    return simplified_points, movement_sequence


if __name__ == "__main__":
    # Example usage:
    input_path = "image.png"
    points, moves = process_image(
        input_path,
        rdp_epsilon=2.0,
        save_debug_prefix="debug",
    )

    print("\nVertical movement sequence:")
    print(moves)

