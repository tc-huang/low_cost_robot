import mujoco
import numpy as np


class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.integration_dt = 0.1
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
        return self.d.qpos[:6].copy()

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def read_ee_pos(self, joint_name='end_effector'):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        joint_id = self.m.body(joint_name).id
        # return self.d.geom_xpos[joint_id]
        return self.d.xpos[joint_id]

    def inverse_kinematics(self, ee_target_pos, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id
        # get the current end effector position
        ee_pos = self.read_ee_pos(joint_name)
        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        # compute target joint velocities
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos)
        # apply the joint velocities
        qpos = self.read_position()
        # q_target_pos = qpos + qdot
        mujoco.mj_integratePos(self.m, qpos, qdot, self.integration_dt)
        return qpos

    def set_target_pos(self, target_pos):
        # print(self.m.jnt_range.T)
        np.clip(target_pos, *self.m.jnt_range.T[:, 0:6], out=target_pos)
        self.d.ctrl = target_pos