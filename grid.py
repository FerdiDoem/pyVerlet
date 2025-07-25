import numpy as np

class HashGrid:
    """Simple spatial hash grid for neighbor search."""

    def __init__(self):
        self.cells = {}
        self.points = None
        self.inv_cell_size = 1.0

    def build(self, points, cell_size):
        """Populate the hash grid with the given points."""
        self.points = np.asarray(points)
        self.inv_cell_size = 1.0 / float(cell_size)
        self.cells = {}
        coords = np.floor(self.points * self.inv_cell_size).astype(int)
        for idx, (gx, gy) in enumerate(coords):
            key = (gx, gy)
            self.cells.setdefault(key, []).append(idx)

    def query_pairs(self, radius):
        """Return index pairs within the given radius."""
        if self.points is None or len(self.points) == 0:
            return np.empty((0, 2), dtype=int)
        cell_range = int(np.ceil(radius * self.inv_cell_size))
        coords = np.floor(self.points * self.inv_cell_size).astype(int)
        pairs = set()
        r2 = radius * radius
        for idx, (cx, cy) in enumerate(coords):
            for dx in range(-cell_range, cell_range + 1):
                for dy in range(-cell_range, cell_range + 1):
                    for j in self.cells.get((cx + dx, cy + dy), []):
                        if j <= idx:
                            continue
                        d = self.points[idx] - self.points[j]
                        if d.dot(d) < r2:
                            pairs.add((idx, j))
        if not pairs:
            return np.empty((0, 2), dtype=int)
        return np.array(sorted(pairs), dtype=int)
