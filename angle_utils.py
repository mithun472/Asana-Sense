import numpy as np
from collections import defaultdict

# Same topology as pose_estimation.py
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

NUM_LANDMARKS = 33


def _build_adjacency():
    adj = defaultdict(set)
    for a, b in POSE_CONNECTIONS:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def build_joint_triplets():
    """
    For each of 33 landmarks, pick 2 neighbors from topology to form
    (neighbor_a, joint, neighbor_b) triplet. Landmarks with <2 neighbors
    (eyes, ears, mouth tips - low degree) get skipped -> no angle possible.
    Priority: pick 2 neighbors giving longest bone-pair (more stable angle)
    if joint has >2 neighbors (e.g. hips/shoulders have 3).
    """
    adj = _build_adjacency()
    triplets = {}
    for j in range(NUM_LANDMARKS):
        neighbors = sorted(adj[j])
        if len(neighbors) < 2:
            continue  # not enough connections to form angle
        if len(neighbors) == 2:
            n1, n2 = neighbors
        else:
            # more than 2 neighbors (e.g. hip/shoulder junctions) - just take first 2
            n1, n2 = neighbors[0], neighbors[1]
        triplets[j] = (n1, j, n2)
    return triplets


JOINT_TRIPLETS = build_joint_triplets()


def calc_angle(a, b, c):
    """Angle at point b, formed by points a-b-c. Uses 2D (x,y) arctan2. Returns degrees 0-180."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba = a - b
    bc = c - b
    cross_2d = ba[0] * bc[1] - ba[1] * bc[0]  # manual scalar cross, np.cross dropped 2D support
    angle = np.degrees(np.arctan2(cross_2d, np.dot(ba, bc)))
    return float(abs(angle))


def calc_all_angles(landmarks):
    """
    landmarks: list/array of 33 (x, y[, z]) points (normalized 0-1 coords fine,
    angle math is scale-invariant).
    Returns dict {landmark_idx: angle_degrees}. Missing joints (low degree) -> None.
    """
    angles = {}
    for j in range(NUM_LANDMARKS):
        if j not in JOINT_TRIPLETS:
            angles[j] = None
            continue
        n1, _, n2 = JOINT_TRIPLETS[j]
        pt_a = (landmarks[n1].x, landmarks[n1].y)
        pt_b = (landmarks[j].x, landmarks[j].y)
        pt_c = (landmarks[n2].x, landmarks[n2].y)
        angles[j] = calc_angle(pt_a, pt_b, pt_c)
    return angles