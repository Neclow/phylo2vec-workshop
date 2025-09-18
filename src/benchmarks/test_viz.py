import phylo2vec as p2v
import pytest

from ..solutions import visualize_tree

VIZ_RANGE = range(10, 60, 10)


# Test the visualize_tree function on vectors
@pytest.mark.parametrize("n_leaves", VIZ_RANGE)
def test_viz_vector(benchmark, n_leaves):
    v = p2v.sample_vector(n_leaves)
    benchmark(visualize_tree, v)


# Test the visualize_tree function on matrices
@pytest.mark.parametrize("n_leaves", VIZ_RANGE)
def test_viz_matrix(benchmark, n_leaves):
    m = p2v.sample_matrix(n_leaves)
    benchmark(visualize_tree, m)
