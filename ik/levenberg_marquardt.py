import mujoco
import numpy as np

#Levenberg-Marquardt method
class LevenbegMarquardtIK:
    
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr, damping):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
        self.damping = damping
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))

    #Levenberg-Marquardt pseudocode implementation
    def calculate(self, goal, init_q, body_id):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        # current_pose = self.data.body(body_id).xpos
        current_pose = self.data.site('TCP').xpos
        error = np.subtract(goal, current_pose)

        while (np.linalg.norm(error) >= self.tol):
            #calculate jacobian
            # mujoco.mj_jac(self.model, self.data, self.jacp, self.jacr, goal, body_id)
            mujoco.mj_jacSite(self.model, self.data, self.jacp, 
                          self.jacr, self.model.site("TCP").id)
            #calculate delta of joint q
            n = self.jacp.shape[1]
            I = np.identity(n)
            product = self.jacp.T @ self.jacp + self.damping * I
            
            if np.isclose(np.linalg.det(product), 0):
                j_inv = np.linalg.pinv(product) @ self.jacp.T
            else:
                j_inv = np.linalg.inv(product) @ self.jacp.T
            
            delta_q = j_inv @ error
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            # error = np.subtract(goal, self.data.body(body_id).xpos)
            error = np.subtract(goal, self.data.site('TCP').xpos)
            print(f"\033[H\033[2Jerror:{error}")