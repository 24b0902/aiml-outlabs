import numpy as np
from typing import Callable, Optional, Tuple, List
from functions.func import func
from .optim import LSLROptimiser


class LSLRAlgo2(LSLROptimiser):
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
        self.batch_size=min(32,self.n_samples)
        H=(1/self.n_samples)* (X.T @ X)
        self.L= np.linalg.norm(H,2)
        self.w_bar = None
        self.full_grad_w_bar = None
        self.inner_counter = 0
        self.inner_steps = self.n_samples
        
        ##



    def lr(self) -> float:
        ## TODO learning rate schedule
        return 1/self.L
        # return 1
    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method
        # w_bar is needed :(
        eta = self.lr()
        if self.inner_counter % self.inner_steps == 0:
            self.w_bar = params.copy()
            self.full_grad_w_bar = self.full_grad(self.w_bar)

        i = np.random.randint(0, self.n_samples)
        xi = self.X[i]
        yi = self.y[i]

        grad_i_w = (xi @ params - yi) * xi

        grad_i_wbar = (xi @ self.w_bar - yi) * xi

        grad = grad_i_w - grad_i_wbar + self.full_grad_w_bar

        params = params - eta * grad

        self.inner_counter += 1
        return params
        # raise NotImplementedError("Implement step method for LSLRAlgo1")
    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2
        lslr=(1/(2*self.n_samples))*np.linalg.norm(self.X @w - self.y)**2
        return lslr
        # raise NotImplementedError("Implement eval_lslr method for LSLRAlgo1")
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        return (1 /( self.n_samples)) * self.X.T @ (self.X @ w - self.y)
        # raise NotImplementedError("Implement full_grad method for LSLRAlgo1")
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
        """
        Compute stochastic gradient G(w, γ) = e_γ * (∇f(w))_γ
        Returns a sparse vector with only the γ-th component non-zero.
        """
        ## TODO Implement stochastic gradient computation
        x_i=self.X[gamma,:]
        y_i=self.y[gamma]
        return x_i*(x_i@w - y_i)
        # raise NotImplementedError("Implement stoch_grad method for LSLRAlgo1")