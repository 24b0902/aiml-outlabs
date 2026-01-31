import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo3(LSLROptimiser):
    """
    Gradient Descent for LSLR with optimal learning rate.
    
    Uses η = 1/L where L is the Lipschitz constant (largest eigenvalue of Hessian).
    For f(w) = (1/2n)||Xw - y||^2, Hessian = (1/n) X^T X
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)
        
        ## TODO Use this for any pre-computations you need
        self.norm_sq=np.linalg.norm(self.X)**2
        self.row_norm_sq=np.linalg.norm(self.X, axis=1)**2
        if self.norm_sq==0:
            self.norm_sq=1e-10
        self.n_samples, self.n_features = X.shape


        ##



    def lr(self) -> float:
        ## TODO learning rate schedule
        return 1/self.norm_sq
    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method
        gamma = np.random.choice(self.n_samples, p=(self.row_norm_sq / self.norm_sq))
        grad=self.stoch_grad(params,gamma)
        x_new=params-self.lr()*grad
        return x_new
        raise NotImplementedError("Implement step method for LSLRAlgo1")
    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2
        lslr=(1/(2*self.n_samples))*np.linalg.norm(self.X@w - self.y)**2
        return lslr
        raise NotImplementedError("Implement eval_lslr method for LSLRAlgo1")
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        grad= (self.X.T @ (self.X @ w - self.y)) / self.n_samples
        return grad
        raise NotImplementedError("Implement full_grad method for LSLRAlgo1")
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
        """
        Compute stochastic gradient G(w, γ) = e_γ * (∇f(w))_γ
        Returns a sparse vector with only the γ-th component non-zero.
        """
        ## TODO Implement stochastic gradient computation
        p_gamma=self.row_norm_sq[gamma]/self.norm_sq
        k_gamma=1.0/p_gamma
        x_i=self.X[gamma,:]
        y_i=self.y[gamma]
        grad= x_i * (np.dot(x_i, w) - y_i) 
        return k_gamma*grad

        raise NotImplementedError("Implement stoch_grad method for LSLRAlgo1")
