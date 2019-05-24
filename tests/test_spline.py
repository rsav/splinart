import splinart as spl
import numpy as np
import pytest

@pytest.mark.parametrize("beg, end", [
    (0 , 1),
    (-2, -1),
    (-20, 10),
])
@pytest.mark.parametrize("nbpoints", [
    (10),
    (20),
    (30)
])
def test_spline(beg, end, nbpoints):
    """
    test that spline return a null vector for a line
    """
    path = spl.line(beg, end, npoints=nbpoints) # create a path from a line
    y2s = spl.spline.spline(path[:, 0], path[:, 1]) # create the spline from the path, nb: path[:,0] is the first column
    assert y2s == pytest.approx(np.zeros(nbpoints)) # check that the second derivative is a null vector

@pytest.mark.parametrize("center, radius", [
    ([0, 0] , 1),
    ([0.5, 0.1] , .1),
    ([-0.5, 0.6] , .3),
])
@pytest.mark.parametrize("nbpoints", [
    (10),
    (20),
    (30)
])
def test_spline_circle(center, radius, nbpoints):
    theta, path = spl.circle(center, radius, npoints=nbpoints)
    y2s = spl.spline.spline(theta, path)
    y_new = np.zeros_like(path)
    spl.spline.splint(theta, path, y2s, theta, y_new)
    assert path == pytest.approx(y_new)