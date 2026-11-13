"""Multilevel force-directed layout of a spanning forest (FM3-style).

The tree is coarsened by grouping each node with its lightest free neighbour,
the coarsest level is laid out from a radial initialisation, then groups are
expanded level by level and refined with Fruchterman-Reingold iterations whose
repulsion is approximated on a quadtree pyramid (Barnes-Hut style).
"""

import numba as nb
import numpy as np


def _tree_csr(n, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order].astype(np.int64)


# --------------------------------------------------------------------------- radial init


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
    """Radial layout of each tree, rooted at its centroid, with subtree-size-weighted wedges."""
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


@nb.njit(cache=True)
def _components(n, indptr, nbrs):
    comp = np.full(n, -1, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        m = _bfs(indptr, nbrs, s, queue, parent)
        for i in range(m):
            comp[queue[i]] = c
        c += 1
    return comp


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


def _radial_layout(n, edges):
    indptr, nbrs = _tree_csr(n, edges)
    x, y, comp = _radial(n, indptr, nbrs)
    x, y = _pack_components(x, y, comp)
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------- forces


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
def _step(pos, force, indptr, nbrs, k, gravity, temp, anchor):
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
        fx -= gravity * (pos[v, 0] - anchor[v, 0])
        fy -= gravity * (pos[v, 1] - anchor[v, 1])
        f = np.sqrt(fx * fx + fy * fy) + 1e-12
        step = min(f, temp)
        new[v, 0] = pos[v, 0] + fx / f * step
        new[v, 1] = pos[v, 1] + fy / f * step
    return new


@nb.njit(cache=True)
def _sort_by_cell(pos, lo, span, g):
    """Finest-grid cell of each particle, and particles grouped by cell (counting sort)."""
    n = pos.shape[0]
    cx = np.empty(n, dtype=np.int64)
    cy = np.empty(n, dtype=np.int64)
    cell_start = np.zeros(g * g + 1, dtype=np.int64)
    for i in range(n):
        a = min(int((pos[i, 0] - lo[0]) / span * g), g - 1)
        b = min(int((pos[i, 1] - lo[1]) / span * g), g - 1)
        cx[i], cy[i] = a, b
        cell_start[a * g + b + 1] += 1
    for c in range(g * g):
        cell_start[c + 1] += cell_start[c]
    fill = cell_start[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = cx[i] * g + cy[i]
        order[fill[c]] = i
        fill[c] += 1
    return cx, cy, order, cell_start


def _force_iterations(pos, indptr, nbrs, k, n_iter, temp0, gravity, anchor):
    n = pos.shape[0]
    # ~1 particle per finest cell over the bounding box: tree layouts leave much of it
    # empty, so occupied cells are denser and the exact near-field term dominates otherwise.
    levels = int(np.clip(np.ceil(np.log2(np.sqrt(max(n, 4)))), 2, 12))
    g = 1 << levels
    offsets = np.array([sum(4**j for j in range(lvl)) for lvl in range(levels + 2)], dtype=np.int64)
    for it in range(n_iter):
        lo = pos.min(axis=0)
        span = (pos.max(axis=0) - lo).max() + 1e-9
        cx, cy, order, cell_start = _sort_by_cell(pos, lo, span, g)
        mass, sx, sy = _build_pyramid(pos, cx, cy, levels, offsets)
        force = _repulsion(pos, cx, cy, levels, offsets, mass, sx, sy, order, cell_start, k * k)
        temp = temp0 * (1 - it / n_iter) + 0.01 * k
        pos = _step(pos, force, indptr, nbrs, k, gravity, temp, anchor)
    return pos


# --------------------------------------------------------------------------- multilevel


@nb.njit(cache=True)
def _coarsen(indptr, nbrs, mass, order):
    """Group each node with its lightest free neighbour, or join the lightest adjacent group.

    Every non-isolated node ends up in a group of size >= 2 (so each level at least halves
    the node count) and groups are connected, so contracting a tree gives a tree.
    """
    n = mass.shape[0]
    group = np.full(n, -1, dtype=np.int64)
    gmass = np.zeros(n)
    n_groups = 0
    for v in order:
        if group[v] != -1:
            continue
        best_free, best_free_m = -1, np.inf
        best_grp, best_grp_m = -1, np.inf
        for p in range(indptr[v], indptr[v + 1]):
            u = nbrs[p]
            if group[u] == -1:
                if mass[u] < best_free_m:
                    best_free, best_free_m = u, mass[u]
            elif gmass[group[u]] < best_grp_m:
                best_grp, best_grp_m = group[u], gmass[group[u]]
        if best_free != -1:
            group[v] = group[best_free] = n_groups
            gmass[n_groups] = mass[v] + mass[best_free]
            n_groups += 1
        elif best_grp != -1:
            group[v] = best_grp
            gmass[best_grp] += mass[v]
        else:
            group[v] = n_groups
            gmass[n_groups] = mass[v]
            n_groups += 1
    return group, gmass[:n_groups].copy(), n_groups


def tree_layout(
    n: int,
    edges: np.ndarray,
    iters_coarse: int = 300,
    iters_fine: int = 50,
    min_nodes: int = 500,
    gravity: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """2D layout of a spanning forest over ``n`` nodes (e.g. from ``minimum_spanning_forest``).

    ``iters_coarse`` force iterations are run on the coarsest level (half as many on
    intermediate levels) and ``iters_fine`` on the full tree. Coarsening stops once a
    level has at most ``min_nodes`` nodes. Results are deterministic for a given ``seed``.

    When a level is expanded, each node is pulled towards its group's position with
    strength ``gravity``. This local gravity keeps branches compact without collapsing
    the global tree structure, unlike a pull towards the centre.

    Returns an ``(n, 2)`` float32 array of coordinates (median edge length ~1).
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rng = np.random.default_rng(seed)
    hierarchy = []  # (group assignment, edges, n) per level, finest first
    cur_edges, cur_n, mass = edges, n, np.ones(n)
    while cur_n > min_nodes:
        indptr, nbrs = _tree_csr(cur_n, cur_edges)
        group, gmass, n_groups = _coarsen(indptr, nbrs, mass, rng.permutation(cur_n))
        if n_groups > 0.8 * cur_n:
            break
        hierarchy.append((group, cur_edges, cur_n))
        ge = group[cur_edges]
        cur_edges = ge[ge[:, 0] != ge[:, 1]]
        cur_n, mass = n_groups, gmass

    # Coarse nodes stand for n / cur_n nodes each: scale k to keep the total area comparable.
    k = np.sqrt(n / cur_n)
    pos = _radial_layout(cur_n, cur_edges)
    pos -= pos.mean(axis=0)
    if len(cur_edges):
        pos *= k / (np.median(np.linalg.norm(pos[cur_edges[:, 0]] - pos[cur_edges[:, 1]], axis=1)) + 1e-12)
    indptr, nbrs = _tree_csr(cur_n, cur_edges)
    # Weak central gravity on the coarsest level only, to keep components together.
    pos = _force_iterations(pos, indptr, nbrs, k, iters_coarse, np.abs(pos).max() / 10 + k, 0.01, np.zeros_like(pos))

    for group, lvl_edges, lvl_n in reversed(hierarchy):
        k = np.sqrt(n / lvl_n)
        anchor = pos[group]
        pos = anchor + rng.normal(scale=0.1 * k, size=(lvl_n, 2))  # place members at their group
        indptr, nbrs = _tree_csr(lvl_n, lvl_edges)
        iters = iters_fine if lvl_n == n else iters_coarse // 2
        pos = _force_iterations(pos, indptr, nbrs, k, iters, 2 * k, gravity, anchor)

    # Repulsion pushes small components far away: re-pack them tightly around the largest one.
    if len(edges) < n - 1:
        comp = _components(n, *_tree_csr(n, edges))
        counts = np.bincount(comp)
        centre = np.stack([np.bincount(comp, pos[:, 0]), np.bincount(comp, pos[:, 1])], axis=1) / counts[:, None]
        rel = pos - centre[comp]
        pos = np.stack(_pack_components(rel[:, 0], rel[:, 1], comp), axis=1)
    return pos.astype(np.float32)
