import numpy as np
from scipy.special import roots_legendre
from collections.abc import Callable


def legendre_integrate(
    f: Callable, n_points: int, args: tuple = None, axis: int = None
):
    """
    Gauss-Legendre quadrature integration
    Performs integration of f over [-inf, inf] using Gauss-Legendre quadrature.
    Interval is split into [-inf, 0], [-1,0], [0,1] and [1, inf].

    Parameters:
    ----------
    f : function
        function to integrate, defined over [-inf, inf]
        Integration argument should be the first positional argument of f
    n_points : int
        number of points for Gauss-Legendre quadrature
    args : tuple, optional
        optional arguments to pass to f

    Returns:
    ---------
    integral: scalar or ndarray
        integral of f over [-inf, inf]
    """
    match args:
        case None:
            _f = f
        case _:
            _f = lambda x: f(x, *args)

    # Compute legendre polynomial roots and associated weights
    points, weights = roots_legendre(n_points)
    x = (1 + points) / 2  # Evaluation points

    # Sample function to check return shape
    _sample = _f(points)
    match _sample:
        case float() | int() | complex():
            shape = None
        case np.ndarray():
            shape = _sample.shape
        case _:
            raise TypeError(
                "Function should return either a numeric value or a numpy array"
            )

    # If no axis was provided while data is multidimensional, assume axis of integration is 0
    if shape and axis is None:
        axis = 0

    # Broadcast arrays to match return shape of the function
    n = len(shape)
    n_left, n_right = axis, n - axis - 1
    _weights = weights[*(n_left * (None,)), :, *(n_right * (None,))]
    _x = x[*(n_left * (None,)), :, *(n_right * (None,))]

    # Compute integrand and perform gauss integration
    integrand = 0.5 * (_f(x) + _f(-x) + 1 / _x**2 * (_f(1 / x) + _f(-1 / x)))
    integral = (integrand * _weights).sum(axis=axis)
    return integral


def main():
    """Test the module computing the integral of a lorentzian over R. Result should be a*pi."""

    def lorentz(w, a):
        return np.outer(1 / (1 + w**2), a)

    n_points = 5
    a = np.arange(5)
    integral = legendre_integrate(lorentz, n_points, args=(a,))
    print(f"Result = {integral}")
    # print(f"Relative error = {abs(integral-a*np.pi)/a*np.pi}")


if __name__ == "__main__":
    main()
