import mujoco
import numpy as np

from ik.ik import IK

# Gradient Descent method
class GradientDescentIK(IK):
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, **kwargs):
        super().__init__(model, data, step_size, tol, alpha, jacp, jacr)

    #Gradient Descent pseudocode implementation
    def calculate_delta_q(self, error):
        #calculate jacobian
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.model.site("TCP").id)
        #calculate gradient
        grad = self.alpha * self.jacp.T @ error
        delta_q = grad
        return delta_q
            