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
        self.n_samples, self.n_features = X.shape
        lipschitz=np.linalg.eigvals((self.X.T @self.X)/self.n_samples)
        self.L=np.max(lipschitz)
        if self.L==0:
            self.L=1e-10
        self.batch_size=min(32,self.n_samples)
        ##



    def lr(self) -> float:
        ## TODO learning rate schedule
        return 1/self.L
    
    def step(self, params: np.ndarray) -> np.ndarray:
        ## TODO Implement the step method
        B=np.random.choice(self.n_samples,self.batch_size,replace=False)
        grad=np.zeros(self.n_features)
        for i in B:
           grad+=self.stoch_grad(params,i)
        grad=grad/self.batch_size
        x_new=params - self.lr()*grad
        return x_new
        raise NotImplementedError("Implement step method for LSLRAlgo1")
    def eval_lslr(self, w: np.ndarray) -> float:
        ## TODO Evaluate LSLR objective: (1/n)||Xw - y||^2
        lslr=(1/(2*self.n_samples))*np.linalg.norm(self.X @w - self.y)**2
        return lslr
        raise NotImplementedError("Implement eval_lslr method for LSLRAlgo1")
    def full_grad(self, w: np.ndarray) -> np.ndarray:
        ## TODO 
        grad= (self.X.T @ (self.X @ w - self.y)) / self.n_samples
        return grad
        raise NotImplementedError("Implement full_grad method for LSLRAlgo1")
    def stoch_grad(self, w: np.ndarray, gamma: int) -> np.ndarray:
       
        ## TODO Implement stochastic gradient computation
        x_i=self.X[gamma,:]
        y_i=self.y[gamma]
        return x_i*(x_i@w - y_i)
        raise NotImplementedError("Implement stoch_grad method for LSLRAlgo1")

