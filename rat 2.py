import heapq
from collections import deque

class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = {i: [] for i in range(n)}

    def add_edge(self, u, v, cost):
        self.adj[u].append((v, cost))
        self.adj[v].append((u, cost))  # undirected graph

    def dfs(self, start, target):
        visited = set()
        stack = [(start, [start], 0)]
        while stack:
            node, path, cost = stack.pop()
            if node == target:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.adj[node]:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], cost + c))
        return None

    def bfs(self, start, target):
        visited = set()
        queue = deque([(start, [start], 0)])
        while queue:
            node, path, cost = queue.popleft()
            if node == target:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.adj[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor], cost + c))
        return None

    def dijkstra(self, start, target):
        heap = [(0, start, [start])]
        visited = set()
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == target:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.adj[node]:
                    if neighbor not in visited:
                        heapq.heappush(heap, (cost + c, neighbor, path + [neighbor]))
        return None

    def a_star(self, start, target, coords):
        def heuristic(u, v):
            (x1, y1), (x2, y2) = coords[u], coords[v]
            return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

        heap = [(0 + heuristic(start, target), 0, start, [start])]
        visited = set()
        while heap:
            est, cost, node, path = heapq.heappop(heap)
            if node == target:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.adj[node]:
                    if neighbor not in visited:
                        g = cost + c
                        f = g + heuristic(neighbor, target)
                        heapq.heappush(heap, (f, g, neighbor, path + [neighbor]))
        return None


# Example usage:
if __name__ == "__main__":
    # Define graph (example)
    g = Graph(5)
    g.add_edge(0, 1, 2)
    g.add_edge(0, 2, 4)
    g.add_edge(1, 2, 1)
    g.add_edge(1, 3, 7)
    g.add_edge(2, 4, 3)

    coords = {0:(0,0), 1:(1,2), 2:(2,1), 3:(3,3), 4:(4,1)}  # for A*

    start, target = 0, 4

    print("DFS:", g.dfs(start, target))
    print("BFS:", g.bfs(start, target))
    print("Dijkstra:", g.dijkstra(start, target))
    print("A*:", g.a_star(start, target, coords))
