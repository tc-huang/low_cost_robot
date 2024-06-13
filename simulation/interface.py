import mujoco
import numpy as np
# from ik.levenberg_marquardt import LevenbegMarquardtIK

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
            yield self.data.qpos
            #compute forward kinematics
            mujoco.mj_forward(self.model, self.data) 
            #calculate new error
            # error = np.subtract(goal, self.data.body(body_id).xpos)
            error = np.subtract(goal, self.data.site('TCP').xpos)
            print(f"\033[H\033[2Jerror:{error}")

class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.m = m
        self.d = d

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        """
        :param pos: numpy array of joint positions in range [-pi, pi]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return (pos / 3.14 + 1.) * 4096

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        """
        :param pwm: numpy array of pwm values in range [0, 4096]
        :return: numpy array of joint positions in range [-pi, pi]
        """
        return (pwm / 2048 - 1) * 3.14

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of pwm values in range [0, 4096]
        :return: numpy array of values in range [0, 1]
        """
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of values in range [0, 1]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return x * 4096

    def read_position(self) -> np.ndarray:
        """
        :return: numpy array of current joint positions in range [0, 4096]
        """
        return self.d.qpos[:6]

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def read_ee_pos(self, joint_name='TCP'):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        if(joint_name=='TCP'):
            return self.d.site('TCP').xpos

        joint_id = self.m.body(joint_name).id
        # return self.d.geom_xpos[joint_id]
        return self.d.xpos[joint_id]

    def inverse_kinematics(self, ee_target_pos, joint_name='TCP'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        #Init variables.
        model = self.m
        data = self.d
        body_id = model.site('TCP').id
        jacp = np.zeros((3, model.nv)) #translation jacobian
        jacr = np.zeros((3, model.nv)) #rotational jacobian
        goal = ee_target_pos
        step_size = 0.5
        tol = 0.01
        alpha = 0.5
        init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        damping = 0.15

        ik = LevenbegMarquardtIK(model, data, step_size, tol, alpha, jacp, jacr, damping)

        #Get desire point
        mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
        for i in ik.calculate(goal, init_q, body_id): #calculate the qpos
            yield i
        return
        result = data.qpos.copy()
        data.qpos = result
        mujoco.mj_forward(model, data)
        return





        if(joint_name=='TCP'):
            joint_id = self.d.site('TCP').id
            # get the current end effector position
            ee_pos = self.d.site('TCP').xpos
        else:
            joint_id = self.m.body(joint_name).id
            # get the current end effector position
            ee_pos = self.d.geom_xpos[joint_id]
        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        # compute target joint velocities
        qpos = self.read_position()
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos)
        # apply the joint velocities
        q_target_pos = qpos + qdot
        return q_target_pos

    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos