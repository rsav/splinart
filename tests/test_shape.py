import splinart as spl
import numpy as np
import pytest
from pytest import approx

def test_circle_1():
    theta, path = spl.circle([0, 0], 1, npoints=2) # create a splinned circle with center [0,0] radius 1 and 2 points
    print(theta)
    assert theta == approx([0, 2*np.pi]) # the resulting vector of theta angles for each point
    #assert path == approx(np.array([[1, 0], [1, 0]]))
    print(path)
    assert path == approx(np.array([[1, 0], [1, 0]])) # and the resulting path must be this

def test_circle_2():
    theta, path = spl.circle([0, 0], 1, npoints=5)

    assert theta == approx(np.linspace(0, 2*np.pi, 5))
    assert path == approx(np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]))

def test_circle_10():
    """
    test with 10 points
    """
    theta, path = spl.circle([0,0],1,npoints=10)

    assert theta == approx(np.linspace(0, 2*np.pi, 10))



def test_line_1():
    path = spl.line(0, 1, npoints=2)

    assert path == approx(np.array([[0, 0.5], [1, 0.5]]))

def test_line_2():
    path = spl.line(0, 1, npoints=3)

    assert path == approx(np.array([[0, 0.5], [0.5, 0.5], [1, 0.5]]))


@pytest.mark.parametrize("nbpoints,expected", [
    (2, np.array([[1, 0], [1, 0]])),
    (5, np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]))
])
def test_circle(nbpoints, expected):
    """
    test with parametrize to run several tests
    """

    theta, path = spl.circle([0, 0], 1, npoints=nbpoints)
    assert path == approx(expected)


@pytest.mark.parametrize("nbpoints,expected_path,expected_theta", [
    (2, np.array([[1, 0], [1, 0]]), np.linspace(0, 2*np.pi, 2)),
    (5, np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]]), np.linspace(0, 2*np.pi, 5))
])
def test_circle2(nbpoints, expected_path, expected_theta):
    """
    test with parametrize to run several tests
    """

    theta, path = spl.circle([0, 0], 1, npoints=nbpoints)
    assert path == approx(expected_path)
    assert theta == approx(expected_theta)




@pytest.mark.parametrize("nbpoints,expected", [
    (2, np.array([[0, 0.5], [1, 0.5]])),
    (3, np.array([[0, 0.5], [0.5, 0.5], [1, 0.5]]))
])
def test_line(nbpoints, expected):
    path = spl.line(0, 1, npoints=nbpoints)
    assert path == approx(expected)




