"""Split a selection over up to 8 workers to minimise (largest load + 2 s per worker).

Durations are whole milliseconds, rounded the way the scorer rounds them, so the cost this
module reports is exactly the scorer's cost for the returned split.
"""

from __future__ import annotations

import heapq
import math

WORKER_COST_MS = 2000
MAX_WORKERS = 8


def duration_ms(seconds: float) -> int:
    return math.floor(seconds * 1000 + 0.5)


def _lpt(items: list[tuple[int, int]], w: int) -> tuple[int, list[list[int]]]:
    heap = [(0, i) for i in range(w)]
    bins: list[list[int]] = [[] for _ in range(w)]
    loads = [0] * w
    for ms, n in items:
        load, i = heapq.heappop(heap)
        bins[i].append(n)
        loads[i] = load + ms
        heapq.heappush(heap, (loads[i], i))
    return max(loads), bins


def pack(tests: list[int], dur_ms: list[int]) -> tuple[int, list[list[int]]]:
    """Return (cost_ms, workers). Empty selection costs 0."""
    if not tests:
        return 0, []
    items = sorted(((dur_ms[n], n) for n in tests), reverse=True)
    total = sum(ms for ms, _ in items)
    biggest = items[0][0]
    best_cost, best_bins = None, None
    for w in range(1, min(MAX_WORKERS, len(items)) + 1):
        # lower bound for this w; stop when even the bound cannot beat the best
        bound = max(biggest, math.ceil(total / w)) + WORKER_COST_MS * w
        if best_cost is not None and bound >= best_cost and w > 1:
            if WORKER_COST_MS * w >= best_cost:
                break
            continue
        makespan, bins = _lpt(items, w)
        bins = [b for b in bins if b]
        cost = makespan + WORKER_COST_MS * len(bins)
        if best_cost is None or cost < best_cost:
            best_cost, best_bins = cost, bins
    return best_cost, best_bins
