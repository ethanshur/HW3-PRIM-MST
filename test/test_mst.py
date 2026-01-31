import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`
    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    # basic considerations
    assert adj_mat.shape == mst.shape, "MST must have same shape as adjacency matrix"
    assert adj_mat.ndim == 2 and adj_mat.shape[0] == adj_mat.shape[1], "adj_mat must be square"

    n = adj_mat.shape[0]

    # check if any weights are negative
    assert np.all(mst >= 0), "MST should not contain negative weights"

    # confirm appropriate edge count
    edge_count = 0
    rows, cols = np.nonzero(mst)
    for i, j in zip(rows, cols):
        if i > j:
            edge_count += 1
    assert edge_count == n - 1, f"MST must contain exactly n-1 edges"

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    adj = np.array([
        [0, 1, 8, 8, 8],
        [1, 0, 1, 8, 8],
        [8, 1, 0, 1, 8],
        [8, 8, 1, 0, 1],
        [8, 8, 8, 1, 0],
    ], dtype=float)

    expected = 4.0

    g = Graph(adj)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, expected)