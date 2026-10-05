"""Gallery matching, burst tiebreak and pending-pool clustering (PLAN.md section 6). Embeddings are L2-normalized,
so cosine distance = 1 - dot product. Brute-force numpy; swap in sqlite-vec behind these functions if it gets big."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class Match:
    character_id: int | None   # None = no confident match
    nearest_character: int | None
    distance: float            # distance to the nearest gallery entry (inf if gallery empty)


def match(embedding: np.ndarray, gallery: np.ndarray, gallery_chars: np.ndarray,
          max_distance: float, k: int) -> Match:
    """Assign if the nearest reference is within max_distance and that character holds the majority
    among the k nearest references that are also within max_distance."""
    if len(gallery) == 0:
        return Match(None, None, float("inf"))
    dist = 1.0 - gallery @ embedding
    order = np.argsort(dist)[:k]
    nearest, d0 = int(gallery_chars[order[0]]), float(dist[order[0]])
    if d0 > max_distance:
        return Match(None, nearest, d0)
    close = [int(gallery_chars[i]) for i in order if dist[i] <= max_distance]
    winner, votes = Counter(close).most_common(1)[0]
    return Match(nearest if winner == nearest and votes * 2 > len(close) else None, nearest, d0)


def burst_assign(m: Match, max_distance: float, margin: float, burst_chars: set[int]) -> int | None:
    """Borderline miss whose nearest character was confidently seen in a nearby-in-time photo."""
    if m.character_id is None and m.nearest_character in burst_chars and m.distance <= max_distance + margin:
        return m.nearest_character
    return None


def dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Cosine DBSCAN. Returns a cluster label per row, -1 for noise. min_samples counts the point itself.
    ponytail: O(n^2) distance matrix; fine for a few thousand pending crops, use an ANN index beyond that."""
    n = len(embeddings)
    labels = np.full(n, -1)
    if n == 0:
        return labels
    neighbors = (1.0 - embeddings @ embeddings.T) <= eps
    core = neighbors.sum(axis=1) >= min_samples
    cluster = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cluster
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for nb in np.flatnonzero(neighbors[j]):
                if labels[nb] == -1:
                    labels[nb] = cluster
                    stack.append(nb)
        cluster += 1
    return labels
