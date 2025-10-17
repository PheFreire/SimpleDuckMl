from simple_duck_ml.activations.i_activation import IActivation
from numpy.typing import NDArray
import numpy as np

class SoftmaxActivation(IActivation):
    name = "softmax"

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_shifted = x - np.max(x, axis=0, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x, axis=0, keepdims=True)

        return exp_x / np.clip(sum_exp, 1e-12, np.inf)

    def derivative(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Retorna o gradiente (jacobiano) do softmax.
        s deve ser o vetor já ativado (saída do softmax).
        Para uso genérico, mas em prática (com cross-entropy)
        o gradiente é (y_pred - y_true).
        """
        # If is vector
        if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
            x = x.reshape(-1, 1)
            jacobian = np.diagflat(x) - np.dot(x, x.T)
            return jacobian

        # If is batch
        batch_size = x.shape[1]
        grads = np.zeros_like(x)
        for i in range(batch_size):
            si = x[:, i:i+1]
            jacobian = np.diagflat(si) - np.dot(si, si.T)
            grads[:, i:i+1] = np.sum(jacobian, axis=1, keepdims=True)
        return grads

