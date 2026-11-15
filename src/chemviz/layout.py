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


@nb.njit(cache=True, parallel=True)
def _build_pyramid(spos, cell_start, levels, offsets, mass, sx, sy):
    """Mass and coordinate sums per cell for each quadtree level (flat storage, filled in place).

    ``spos`` holds positions grouped by finest cell (see ``_sort_by_cell``): the finest
    level sums each cell's contiguous slice, coarser levels sum their 4 children.
    """
    g = 1 << levels
    base = offsets[levels]
    for c in nb.prange(g * g):
        x, y = 0.0, 0.0
        for p in range(cell_start[c], cell_start[c + 1]):
            x += spos[p, 0]
            y += spos[p, 1]
        mass[base + c] = cell_start[c + 1] - cell_start[c]
        sx[base + c] = x
        sy[base + c] = y
    for lvl in range(levels - 1, -1, -1):
        gl = 1 << lvl
        child_base, gc = offsets[lvl + 1], gl * 2
        for c in nb.prange(gl * gl):
            a, b = c // gl, c % gl
            m, x, y = 0.0, 0.0, 0.0
            for da in range(2):
                for db in range(2):
                    child = child_base + (2 * a + da) * gc + (2 * b + db)
                    m += mass[child]
                    x += sx[child]
                    y += sy[child]
            mass[offsets[lvl] + c] = m
            sx[offsets[lvl] + c] = x
            sy[offsets[lvl] + c] = y


@nb.njit(cache=True, parallel=True)
def _repulsion(spos, levels, offsets, mass, sx, sy, cell_start, k2, order, force):
    """Repulsive forces, written to ``force`` in original order.

    Inputs are in cell order (``_sort_by_cell``). All particles of a finest cell share
    their far-field interaction lists, so the coarse levels (all but the finest) are
    evaluated once per cell at its centroid, as a force and its Jacobian, and extended
    to each particle by a first-order Taylor expansion. The finest far-field level and
    the near field (3x3 neighbouring finest cells, exact) are computed per particle.
    """
    g_fine = 1 << levels
    for c in nb.prange(g_fine * g_fine):
        start, stop = cell_start[c], cell_start[c + 1]
        if start == stop:
            continue
        ci, cj = c // g_fine, c % g_fine
        x0, y0 = sx[offsets[levels] + c] / (stop - start), sy[offsets[levels] + c] / (stop - start)
        fx0, fy0, jxx, jxy = 0.0, 0.0, 0.0, 0.0  # force at the centroid; Jyy = -Jxx (harmonic field)
        for lvl in range(2, levels):
            shift = levels - lvl
            g = 1 << lvl
            li, lj = ci >> shift, cj >> shift
            pi, pj = li >> 1, lj >> 1
            for a in range(max(0, 2 * (pi - 1)), min(g, 2 * (pi + 2))):
                for b in range(max(0, 2 * (pj - 1)), min(g, 2 * (pj + 2))):
                    if abs(a - li) <= 1 and abs(b - lj) <= 1:
                        continue
                    cell = offsets[lvl] + a * g + b
                    m = mass[cell]
                    if m == 0.0:
                        continue
                    dx = x0 - sx[cell] / m
                    dy = y0 - sy[cell] / m
                    inv = 1.0 / (dx * dx + dy * dy + 1e-9)
                    w = m * k2 * inv
                    fx0 += w * dx
                    fy0 += w * dy
                    jxx += w * (1.0 - 2.0 * dx * dx * inv)
                    jxy -= w * 2.0 * dx * dy * inv
        pi, pj = ci >> 1, cj >> 1
        for i in range(start, stop):
            xi, yi = spos[i, 0], spos[i, 1]
            ox, oy = xi - x0, yi - y0
            fx = fx0 + jxx * ox + jxy * oy
            fy = fy0 + jxy * ox - jxx * oy
            # Finest far-field level, per particle.
            for a in range(max(0, 2 * (pi - 1)), min(g_fine, 2 * (pi + 2))):
                for b in range(max(0, 2 * (pj - 1)), min(g_fine, 2 * (pj + 2))):
                    if abs(a - ci) <= 1 and abs(b - cj) <= 1:
                        continue
                    cell = a * g_fine + b
                    m = mass[offsets[levels] + cell]
                    if m == 0.0:
                        continue
                    dx = xi - sx[offsets[levels] + cell] / m
                    dy = yi - sy[offsets[levels] + cell] / m
                    d2 = dx * dx + dy * dy + 1e-9
                    fx += m * k2 * dx / d2
                    fy += m * k2 * dy / d2
            # Near field: exact interactions with particles in the 3x3 neighbouring finest cells.
            for a in range(max(0, ci - 1), min(g_fine, ci + 2)):
                for b in range(max(0, cj - 1), min(g_fine, cj + 2)):
                    cell = a * g_fine + b
                    for j in range(cell_start[cell], cell_start[cell + 1]):
                        if j == i:
                            continue
                        dx = xi - spos[j, 0]
                        dy = yi - spos[j, 1]
                        d2 = dx * dx + dy * dy + 1e-9
                        fx += k2 * dx / d2
                        fy += k2 * dy / d2
            force[order[i], 0] = fx
            force[order[i], 1] = fy


@nb.njit(cache=True, parallel=True)
def _step(pos, force, indptr, nbrs, k, gravity, temp, anchor, new):
    n = pos.shape[0]
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


@nb.njit(cache=True, parallel=True)
def _sort_by_cell(pos, g, cell, slot, cell_start, order, spos):
    """Group particles by finest-grid cell (counting sort), filling the given buffers.

    Gives positions in cell order, the permutation ``order``
    (sorted index -> original index) and the CSR-style ``cell_start`` offsets.
    """
    n = pos.shape[0]
    lo0, lo1, hi0, hi1 = pos[0, 0], pos[0, 1], pos[0, 0], pos[0, 1]
    for i in range(n):
        lo0, hi0 = min(lo0, pos[i, 0]), max(hi0, pos[i, 0])
        lo1, hi1 = min(lo1, pos[i, 1]), max(hi1, pos[i, 1])
    span = max(hi0 - lo0, hi1 - lo1) + 1e-9
    for i in nb.prange(n):
        a = min(int((pos[i, 0] - lo0) / span * g), g - 1)
        b = min(int((pos[i, 1] - lo1) / span * g), g - 1)
        cell[i] = a * g + b
    cell_start[:] = 0
    for i in range(n):
        cell_start[cell[i] + 1] += 1
    for c in range(g * g):
        cell_start[c + 1] += cell_start[c]
    # Only the slot assignment is sequential (it keeps the sort stable); the scatter is parallel.
    for i in range(n):
        slot[i] = cell_start[cell[i]]
        cell_start[cell[i]] += 1
    for c in range(g * g, 0, -1):
        cell_start[c] = cell_start[c - 1]
    cell_start[0] = 0
    for i in nb.prange(n):
        p = slot[i]
        order[p] = i
        spos[p, 0], spos[p, 1] = pos[i, 0], pos[i, 1]


@nb.njit(cache=True)
def _iterate(pos, indptr, nbrs, k, n_iter, temp0, gravity, anchor, levels, offsets):
    n = pos.shape[0]
    g = 1 << levels
    cell = np.empty(n, dtype=np.int64)
    slot = np.empty(n, dtype=np.int64)
    cell_start = np.empty(g * g + 1, dtype=np.int64)
    order = np.empty(n, dtype=np.int64)
    spos = np.empty_like(pos)
    mass = np.empty(offsets[-1])
    sx = np.empty(offsets[-1])
    sy = np.empty(offsets[-1])
    force = np.empty_like(pos)
    new = np.empty_like(pos)
    for it in range(n_iter):
        _sort_by_cell(pos, g, cell, slot, cell_start, order, spos)
        _build_pyramid(spos, cell_start, levels, offsets, mass, sx, sy)
        _repulsion(spos, levels, offsets, mass, sx, sy, cell_start, k * k, order, force)
        temp = temp0 * (1 - it / n_iter) + 0.01 * k
        _step(pos, force, indptr, nbrs, k, gravity, temp, anchor, new)
        pos, new = new, pos
    return pos


def _force_iterations(pos, indptr, nbrs, k, n_iter, temp0, gravity, anchor):
    n = pos.shape[0]
    # ~1 particle per finest cell over the bounding box: tree layouts leave much of it
    # empty, so occupied cells are denser and the exact near-field term dominates otherwise.
    levels = int(np.clip(np.ceil(np.log2(np.sqrt(max(n, 4)))), 2, 12))
    offsets = np.array([sum(4**j for j in range(lvl)) for lvl in range(levels + 2)], dtype=np.int64)
    # Each iteration launches several parallel regions, which costs from microseconds to
    # milliseconds each (far more on a loaded machine): small levels use fewer threads.
    max_threads = nb.get_num_threads()
    nb.set_num_threads(int(np.clip(n // 5000, 1, max_threads)))
    try:
        return _iterate(np.array(pos, dtype=np.float64), indptr, nbrs, float(k), n_iter,
                        float(temp0), float(gravity), np.ascontiguousarray(anchor, dtype=np.float64),
                        levels, offsets)
    finally:
        nb.set_num_threads(max_threads)


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
