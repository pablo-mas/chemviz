"""Candidate tree layouts for the TMAP reimplementation (prototypes, not public API).

1. radial_layout: subtree-size-weighted radial layout, O(n), deterministic.
2. force_layout: Fruchterman-Reingold on the tree, initialised from the radial layout,
   with repulsion approximated by a quadtree pyramid (Barnes-Hut style).
"""

import numba as nb
import numpy as np


def tree_csr(n, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order].astype(np.int64)


# --------------------------------------------------------------------------- radial


@nb.njit(cache=True)
def _bfs(indptr, nbrs, root, queue, parent):
    queue[0] = root
    parent[root] = root
    head, tail = 0, 1
    while head < tail:
        v = queue[head]
        head += 1
        for p in range(indptr[v], indptr[v + 1]):
            u = nbrs[p]
            if parent[u] == -1:
                parent[u] = v
                queue[tail] = u
                tail += 1
    return tail


@nb.njit(cache=True)
def _subtree_sizes(queue, m, parent, size):
    for i in range(m):
        size[queue[i]] = 1
    for i in range(m - 1, 0, -1):
        v = queue[i]
        size[parent[v]] += size[v]


@nb.njit(cache=True)
def _radial(n, indptr, nbrs):
    x = np.zeros(n)
    y = np.zeros(n)
    comp = np.full(n, -1, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    size = np.zeros(n, dtype=np.int64)
    wstart = np.zeros(n)
    wwidth = np.zeros(n)
    depth = np.zeros(n)
    queue = np.empty(n, dtype=np.int64)
    n_comp = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        m = _bfs(indptr, nbrs, s, queue, parent)
        _subtree_sizes(queue, m, parent, size)
        # Root at the centroid so the layout is balanced.
        best, best_part = s, m + 1
        for i in range(m):
            v = queue[i]
            part = m - size[v]
            for p in range(indptr[v], indptr[v + 1]):
                u = nbrs[p]
                if parent[u] == v and u != v:
                    part = max(part, size[u])
            if part < best_part:
                best, best_part = v, part
        for i in range(m):
            v = queue[i]
            parent[v] = -1
            comp[v] = n_comp
        m = _bfs(indptr, nbrs, best, queue, parent)
        _subtree_sizes(queue, m, parent, size)
        wstart[best], wwidth[best], depth[best] = 0.0, 2 * np.pi, 0.0
        for i in range(m):
            v = queue[i]
            total = size[v] - 1
            cursor = wstart[v]
            for p in range(indptr[v], indptr[v + 1]):
                u = nbrs[p]
                if parent[u] != v or u == v:
                    continue
                w = wwidth[v] * size[u] / total
                wstart[u], wwidth[u] = cursor, w
                cursor += w
                depth[u] = depth[v] + 1
                angle = wstart[u] + w / 2
                x[u] = depth[u] * np.cos(angle)
                y[u] = depth[u] * np.sin(angle)
        n_comp += 1
    return x, y, comp


def _pack_components(x, y, comp):
    """Shelf-pack components (largest first) into a roughly square area."""
    n_comp = comp.max() + 1
    sizes = np.bincount(comp, minlength=n_comp)
    radius = np.zeros(n_comp)
    np.maximum.at(radius, comp, np.hypot(x, y))
    radius += 1.0
    order = np.argsort(-sizes, kind="stable")
    width = np.sqrt(((2 * radius) ** 2).sum())
    cx, cy = np.zeros(n_comp), np.zeros(n_comp)
    px, py, row_h = 0.0, 0.0, 0.0
    for c in order:
        d = 2 * radius[c]
        if px + d > width and px > 0:
            px, py, row_h = 0.0, py - row_h, 0.0
        cx[c], cy[c] = px + radius[c], py - radius[c]
        px += d
        row_h = max(row_h, d)
    return x + cx[comp], y + cy[comp]


def radial_layout(n, edges):
    indptr, nbrs = tree_csr(n, edges)
    x, y, comp = _radial(n, indptr, nbrs)
    x, y = _pack_components(x, y, comp)
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------- force


@nb.njit(cache=True)
def _build_pyramid(pos, cx, cy, levels, offsets):
    """Mass and coordinate sums per cell for each quadtree level (flat storage)."""
    total = offsets[-1]
    mass = np.zeros(total)
    sx = np.zeros(total)
    sy = np.zeros(total)
    for i in range(cx.shape[0]):
        for lvl in range(levels + 1):
            shift = levels - lvl
            g = 1 << lvl
            cell = offsets[lvl] + (cx[i] >> shift) * g + (cy[i] >> shift)
            mass[cell] += 1.0
            sx[cell] += pos[i, 0]
            sy[cell] += pos[i, 1]
    return mass, sx, sy


@nb.njit(cache=True, parallel=True)
def _repulsion(pos, cx, cy, levels, offsets, mass, sx, sy, order, cell_start, k2):
    n = pos.shape[0]
    g_fine = 1 << levels
    force = np.zeros((n, 2))
    for i in nb.prange(n):
        xi, yi = pos[i, 0], pos[i, 1]
        fx, fy = 0.0, 0.0
        # Far field: cells that are children of the parent's neighbours but not neighbours themselves.
        for lvl in range(2, levels + 1):
            shift = levels - lvl
            g = 1 << lvl
            ci, cj = cx[i] >> shift, cy[i] >> shift
            pi, pj = ci >> 1, cj >> 1
            for a in range(max(0, 2 * (pi - 1)), min(g, 2 * (pi + 2))):
                for b in range(max(0, 2 * (pj - 1)), min(g, 2 * (pj + 2))):
                    if abs(a - ci) <= 1 and abs(b - cj) <= 1:
                        continue
                    cell = offsets[lvl] + a * g + b
                    m = mass[cell]
                    if m == 0.0:
                        continue
                    dx = xi - sx[cell] / m
                    dy = yi - sy[cell] / m
                    d2 = dx * dx + dy * dy + 1e-9
                    fx += m * k2 * dx / d2
                    fy += m * k2 * dy / d2
        # Near field: exact interactions with particles in the 3x3 neighbouring finest cells.
        ci, cj = cx[i], cy[i]
        for a in range(max(0, ci - 1), min(g_fine, ci + 2)):
            for b in range(max(0, cj - 1), min(g_fine, cj + 2)):
                c = a * g_fine + b
                for p in range(cell_start[c], cell_start[c + 1]):
                    j = order[p]
                    if j == i:
                        continue
                    dx = xi - pos[j, 0]
                    dy = yi - pos[j, 1]
                    d2 = dx * dx + dy * dy + 1e-9
                    fx += k2 * dx / d2
                    fy += k2 * dy / d2
        force[i, 0] = fx
        force[i, 1] = fy
    return force


@nb.njit(cache=True, parallel=True)
def _step(pos, force, indptr, nbrs, k, gravity, temp):
    n = pos.shape[0]
    new = np.empty_like(pos)
    for v in nb.prange(n):
        fx, fy = force[v, 0], force[v, 1]
        for p in range(indptr[v], indptr[v + 1]):
            u = nbrs[p]
            dx = pos[u, 0] - pos[v, 0]
            dy = pos[u, 1] - pos[v, 1]
            d = np.sqrt(dx * dx + dy * dy) + 1e-9
            fx += d * dx / k  # FR attraction d^2/k along the unit vector
            fy += d * dy / k
        fx -= gravity * pos[v, 0]
        fy -= gravity * pos[v, 1]
        f = np.sqrt(fx * fx + fy * fy) + 1e-12
        step = min(f, temp)
        new[v, 0] = pos[v, 0] + fx / f * step
        new[v, 1] = pos[v, 1] + fy / f * step
    return new


def force_layout(n, edges, n_iter=200, init=None, gravity=0.01):
    indptr, nbrs = tree_csr(n, edges)
    pos = radial_layout(n, edges) if init is None else init.copy()
    pos -= pos.mean(axis=0)
    # Scale so the median edge length is 1 (= k).
    pos /= np.median(np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)) + 1e-12
    k = 1.0
    levels = int(np.clip(np.ceil(np.log2(np.sqrt(max(n, 4) / 4))), 2, 11))
    g = 1 << levels
    offsets = np.array([sum(4**j for j in range(lvl)) for lvl in range(levels + 2)], dtype=np.int64)
    temp0 = np.abs(pos).max() / 10
    for it in range(n_iter):
        lo = pos.min(axis=0)
        span = (pos.max(axis=0) - lo).max() + 1e-9
        cell = np.minimum(((pos - lo) / span * g).astype(np.int64), g - 1)
        cx, cy = cell[:, 0].copy(), cell[:, 1].copy()
        mass, sx, sy = _build_pyramid(pos, cx, cy, levels, offsets)
        fine = cx * g + cy
        order = np.argsort(fine, kind="stable")
        cell_start = np.zeros(g * g + 1, dtype=np.int64)
        np.cumsum(np.bincount(fine, minlength=g * g), out=cell_start[1:])
        force = _repulsion(pos, cx, cy, levels, offsets, mass, sx, sy, order, cell_start, k * k)
        temp = temp0 * (1 - it / n_iter) + 0.01
        pos = _step(pos, force, indptr, nbrs, k, gravity, temp)
    return pos
