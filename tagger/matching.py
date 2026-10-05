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


def split_conflicts(embeddings: np.ndarray, groups: np.ndarray, labels: np.ndarray, min_samples: int,
                    eps: float | None = None) -> np.ndarray:
    """One photo can't show the same character twice. A cluster holding two detections from the same photo
    (`groups` = asset id per row) chained two characters together: split it 2-means style, seeded by the
    most distant same-photo pair (pinned to opposite sides), until no cluster has a conflict.
    Clusters that end up smaller than min_samples become noise."""
    labels = labels.copy()
    next_label = labels.max() + 1 if len(labels) else 0
    queue = [lab for lab in set(labels.tolist()) if lab != -1]
    while queue:
        lab = queue.pop()
        idx = np.flatnonzero(labels == lab)
        best = None
        for g in set(groups[idx].tolist()):
            members = idx[groups[idx] == g]
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    d = 1.0 - float(embeddings[members[a]] @ embeddings[members[b]])
                    if best is None or d > best[0]:
                        best = (d, members[a], members[b])
        if best is None:
            if len(idx) < min_samples:
                labels[idx] = -1
            continue
        _, i, j = best
        seeds = np.stack([embeddings[i], embeddings[j]])
        for _ in range(10):
            side = np.argmax(embeddings[idx] @ seeds.T, axis=1)
            side[idx == i], side[idx == j] = 0, 1
            new = np.stack([embeddings[idx[side == s]].mean(axis=0) for s in (0, 1)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, seeds):
                break
            seeds = new
        labels[idx[side == 1]] = next_label
        queue += [lab, next_label]
        next_label += 1
    if eps is None:
        return labels
    # 2-means sides are not density-based: a point can end up with no neighbour on its side. Re-run DBSCAN inside
    # every final cluster so each member is again within eps of a core point; leftovers become noise.
    out = np.full_like(labels, -1)
    nxt = 0
    for lab in sorted(set(labels.tolist()) - {-1}):
        idx = np.flatnonzero(labels == lab)
        sub = dbscan(embeddings[idx], eps, min_samples)
        for s in sorted(set(sub.tolist()) - {-1}):
            out[idx[sub == s]] = nxt
            nxt += 1
    return out


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
