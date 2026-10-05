import numpy as np

from tagger.matching import Match, burst_assign, dbscan, match


def unit(*v):
    a = np.array(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def near(base, eps, seed):
    """A unit vector at roughly cosine distance eps from base."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=base.shape).astype(np.float32)
    noise -= noise @ base * base
    noise /= np.linalg.norm(noise)
    theta = np.arccos(1 - eps)
    return (np.cos(theta) * base + np.sin(theta) * noise).astype(np.float32)


A = unit(1, 0, 0, 0)
B = unit(0, 1, 0, 0)


def test_match_assigns_nearest_within_distance():
    gallery = np.stack([near(A, 0.05, 1), near(A, 0.06, 2), near(B, 0.05, 3)])
    m = match(near(A, 0.03, 4), gallery, np.array([1, 1, 2]), max_distance=0.15, k=5)
    assert m.character_id == 1 and m.distance < 0.15


def test_match_rejects_far_and_empty():
    gallery = np.stack([near(A, 0.05, 1)])
    m = match(B, gallery, np.array([1]), 0.15, 5)
    assert m.character_id is None and m.nearest_character == 1 and m.distance > 0.15
    assert match(A, np.empty((0, 4), np.float32), np.empty(0, int), 0.15, 5).character_id is None


def test_match_requires_majority_among_close_neighbours():
    q = A
    gallery = np.stack([near(A, 0.02, 1), near(A, 0.03, 2), near(A, 0.04, 3)])
    # Nearest is character 1 but characters 2 hold the majority of close neighbours: ambiguous, no assignment.
    m = match(q, gallery, np.array([1, 2, 2]), 0.15, 5)
    assert m.character_id is None and m.nearest_character == 1


def test_burst_only_rescues_borderline_with_nearby_character():
    borderline = Match(None, 7, 0.17)
    assert burst_assign(borderline, 0.15, 0.05, {7}) == 7
    assert burst_assign(borderline, 0.15, 0.05, {8}) is None
    assert burst_assign(Match(None, 7, 0.25), 0.15, 0.05, {7}) is None
    assert burst_assign(Match(7, 7, 0.05), 0.15, 0.05, {7}) is None  # already matched, not a burst


def test_dbscan_needs_min_samples_and_separates_clusters():
    emb = np.stack([near(A, 0.03, i) for i in range(3)] + [near(B, 0.03, 10 + i) for i in range(2)] + [unit(0, 0, 1, 0)])
    labels = dbscan(emb, eps=0.15, min_samples=3)
    assert len(set(labels[:3])) == 1 and labels[0] != -1
    assert list(labels[3:]) == [-1, -1, -1]  # B has only 2 faces, the last point is alone


def test_split_conflicts_separates_two_characters_chained_in_one_cluster():
    from tagger.matching import split_conflicts

    # Two suits photographed together: each photo has one A head and one B head, DBSCAN chained them into one cluster.
    emb = np.stack([near(A, 0.03, i) for i in range(4)] + [near(B, 0.03, 10 + i) for i in range(4)])
    groups = np.array(["p1", "p2", "p3", "p4", "p1", "p2", "p3", "p4"])
    labels = split_conflicts(emb, groups, np.zeros(8, dtype=int), min_samples=3)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]
    # Too small after the split -> noise.
    labels = split_conflicts(emb[[0, 1, 4, 5]], groups[[0, 1, 4, 5]], np.zeros(4, dtype=int), min_samples=3)
    assert list(labels) == [-1, -1, -1, -1]


def test_split_conflicts_density_recheck_drops_stranded_points():
    from tagger.matching import split_conflicts

    # A and B chained by one bridge point X that sits between them; after splitting, X has no eps-neighbour.
    x = (A + B) / np.linalg.norm(A + B)
    emb = np.stack([near(A, 0.02, i) for i in range(3)] + [near(B, 0.02, 10 + i) for i in range(3)] + [x])
    groups = np.array(["p1", "p2", "p3", "p1", "p2", "p3", "p9"])
    labels = split_conflicts(emb, groups, np.zeros(7, dtype=int), min_samples=3, eps=0.15)
    assert labels[6] == -1 and labels[0] != labels[3] and -1 not in labels[:6]
