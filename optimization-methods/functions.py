import numpy as np
import numpy.typing as npt

class Quadratic:
    """
    f(x) = (1/2) x^T A x + b^T x + c
    """
    def __init__(self, A: npt.NDArray[np.float64], b: npt.NDArray[np.float64], c: npt.NDArray[np.float64]):
        """
        Args:
            - A: (2, 2)
            - b: (2, 1)
            - c: (1, 1)
        """
        self.A = A
        self.b = b
        self.c = c
    
    def forward(self, x: npt.NDArray[np.float64]):
        """
        Args:
            - x: (2, 1)
        """
        value = 0.5*(x.T @ self.A @ x) + self.b.T @ x + self.c
        return value

    def gradient(self, x: npt.NDArray[np.float64]):
        """
        Args:
            - x: (2, 1)
        """
        gradient = self.A @ x + self.b
        return gradient

    def hessian(self, x: npt.NDArray[np.float64]):
        """
        Args:
            - x: (2, 1)
        """
        hessian = self.A
        return hessian

class Trid:
    """
    f(x) = sum{i=1 to n}(xi - 1)**2 - sum{i=2 to n}(x(i-1) x(i))
    """   

if __name__ == "__main__":
    A = np.array([[2, 0], [0, 2]])    # (2, 2)
    b = np.array([2, 2]).reshape(-1, 1)    # (2, 1)
    c = np.array([-10]).reshape(-1, 1)    # (1, 1)

    quadratic_fn = Quadratic(A, b, c)

    x = np.array([1, -5]).reshape(-1, 1)    # (2, 1)

    value = quadratic_fn.forward(x)
    print(value, value.shape)

    gradient = quadratic_fn.gradient(x)
    print(gradient, gradient.shape)

    hessian = quadratic_fn.hessian(x)
    print(hessian, hessian.shape)