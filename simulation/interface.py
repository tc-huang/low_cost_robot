import mujoco
import numpy as np
from ik.gauss_newton import GaussNewtonIK

import mujoco
import numpy as np

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
        step_size = 1.0
        tol = 0.001
        alpha = 0.5
        # init_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        init_q = self.read_position()

        ik = GaussNewtonIK(model, data, step_size, tol, alpha, jacp, jacr)

        #Get desire point
        # mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
        for item in ik.calculate(goal, init_q, 200): #calculate the qpos
            yield item
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