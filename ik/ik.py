import mujoco
import numpy as np
from abc import ABC, abstractmethod

class IK(ABC):
    def __init__(self, model, data, step_size, tol, alpha, jacp, jacr):
        self.model = model
        self.data = data
        self.step_size = step_size
        self.tol = tol
        self.alpha = alpha
        self.jacp = jacp
        self.jacr = jacr
    
    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""
        for i in range(len(q)):
            q[i] = max(self.model.jnt_range[i][0], 
                       min(q[i], self.model.jnt_range[i][1]))
    
    @abstractmethod
    def calculate_delta_q(self, error):
        pass
    
    def calculate(self, goal, init_q, limit_iter):
        """Calculate the desire joints angles for goal"""
        self.data.qpos = init_q
        mujoco.mj_forward(self.model, self.data)
        # current_pose = self.data.body(body_id).xpos
        current_pose = self.data.site('TCP').xpos
        error = np.subtract(goal, current_pose)
        
        iterations = 0

        while (np.linalg.norm(error) >= self.tol):
            if(iterations >= limit_iter):
                break
            delta_q = self.calculate_delta_q(error)
            #compute next step
            self.data.qpos += self.step_size * delta_q
            #check limits
            self.check_joint_limits(self.data.qpos)
            yield self.data.qpos
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            # error = np.subtract(goal, self.data.body(body_id).xpos)
            error = np.subtract(goal, self.data.site('TCP').xpos)
            # print(f"\033[H\033[2Jerror:{error}")
            iterations += 1
        return np.linalg.norm(error), iterations