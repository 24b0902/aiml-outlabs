import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo1(LSLROptimiser):
    """
    Gradient Descent for LSLR with optimal learning rate.
    
    Uses η = 1/L where L is the Lipschitz constant (largest eigenvalue of Hessian).
    For f(w) = (1/2n)||Xw - y||^2, Hessian = (1/n) X^T X
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y)
        
        ## TODO Use this for any pre-computations you need

        self.X=X
        self.y=y
        self.n_samples, self.n_features = X.shape
        hes=(1/self.n_samples)* ( self.X.T @ self.X )
        self.L=np.linalg.norm(hes,2)


        ##



    def lr(self) -> float:
        ## TODO learning rate schedule
        lr=1/self.L
        return lr
    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method
        eta=self.lr()
        gamma = np.random.randint(0, self.n_features)
        return params- eta*self.stoch_grad(params, gamma)
        # raise NotImplementedError("Implement step method for LSLRAlgo1")
    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2
        residual = self.X @ w - self.y
        return (1 / self.n_samples) * (residual @ residual)
        # raise NotImplementedError("Implement eval_lslr method for LSLRAlgo1")
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        return (2/self.n_samples) * self.X.T @ (self.X @ w - self.y)
        # raise NotImplementedError("Implement full_grad method for LSLRAlgo1")
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
       
        ## TODO Implement stochastic gradient computation
        # e_gamma=np.zeros(w.size[0])
        d = self.X.shape[1]
        grad = self.full_grad(w)        
        G = np.zeros_like(w)
        G[gamma] = d * grad[gamma]
        
        return G

        # raise NotImplementedError("Implement stoch_grad method for LSLRAlgo1")