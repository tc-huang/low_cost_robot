import mujoco
import numpy as np

from ik.ik import IK

#Levenberg-Marquardt method
class LevenbegMarquardtIK(IK):
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        super().__init__(model, data, step_size, tol, alpha, jacp, jacr)
        self.damping = damping
    
    #Levenberg-Marquardt pseudocode implementation
    def calculate_delta_q(self, error):
        #calculate jacobian
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.model.site("TCP").id)
        #calculate delta of joint q
        n = self.jacp.shape[1]
        I = np.identity(n)
        product = self.jacp.T @ self.jacp + self.damping * I
        
        if np.isclose(np.linalg.det(product), 0):
            j_inv = np.linalg.pinv(product) @ self.jacp.T
        else:
            j_inv = np.linalg.inv(product) @ self.jacp.T
        
        delta_q = j_inv @ error
        return delta_q