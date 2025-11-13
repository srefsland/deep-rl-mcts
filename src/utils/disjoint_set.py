class DisjointSet:
    def __init__(self, elements=None):
        self.parent = {}
        self.rank = {}
        if elements is not None:
            for elem in elements:
                self.parent[elem] = elem
                self.rank[elem] = 0

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        xroot = self.find(x)
        yroot = self.find(y)
        if xroot == yroot:
            return

        if self.rank[xroot] < self.rank[yroot]:
            self.parent[xroot] = yroot
        elif self.rank[xroot] > self.rank[yroot]:
            self.parent[yroot] = xroot
        else:
            self.parent[yroot] = xroot
            self.rank[xroot] += 1

    def copy(self):
        new_ds = DisjointSet()
        new_ds.parent = self.parent.copy()
        new_ds.rank = self.rank.copy()
        return new_ds

    def __contains__(self, x):
        return x in self.parent

    def __repr__(self):
        return f"DisjointSet({list(self.parent.keys())})"
