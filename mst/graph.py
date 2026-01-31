import numpy as np
from typing import Union
import heapq

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """

        adj = self.adj_mat
        n = adj.shape[0]

        self.mst = np.zeros_like(adj, dtype=float)

        visited = [False] * n
        start = 0
        visited[start] = True

        # heap entries are (weight, from_node, to_node)
        heap = []

        # push edges from the starting node
        for v in range(n):
            w = adj[start, v]
            if w != 0:
                heapq.heappush(heap, (w, start, v))

        while heap and not all(visited):
            w, u, v = heapq.heappop(heap)

            # skip edges leading to already-visited vertices
            if visited[v]:
                continue

            self.mst[u, v] = w
            self.mst[v, u] = w

            visited[v] = True

            # add all outgoing edges from v to unvisited nodes
            for nxt in range(n):
                w2 = adj[v, nxt]
                if not visited[nxt] and w2 != 0:
                    heapq.heappush(heap, (w2, v, nxt))
