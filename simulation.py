import time
import mujoco
import mujoco.viewer
import numpy as np
from simulation.plot import RobotPlot
from simulation.interface import SimulatedRobot

m = mujoco.MjModel.from_xml_path('simulation/low_cost_robot/scene.xml')
d = mujoco.MjData(m)

r = SimulatedRobot(m, d)
robotPlot = RobotPlot(r, period=m.opt.timestep, frames=500, length=3)

# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 0.01

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-10

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Maximum allowable joint velocity in rad/s. Set to 0 to disable.
max_angvel = 0.0

# Define a trajectory for the end-effector site to follow.
class Circle:
  def __init__(self, center: np.ndarray, point: np.ndarray, freq: float):
    diff = point[0:2] - center[0:2]
    x, y = diff

    self.center = center
    self.theta  = np.arctan2([y], [x])[0]  # compute the angle
    self.omega  = 2 * np.pi * freq
    self.radius = np.linalg.norm(diff)
    print(self.center, self.theta, self.omega, self.radius)

  def __call__(self, t):
    angle = self.omega * t + self.theta
    ret = self.radius * np.array([np.cos(angle), np.sin(angle)]) + self.center
    # print(ret)
    return ret

def mse(x, y):
  return np.mean((np.array(x) - np.array(y)) ** 2)

with mujoco.viewer.launch_passive(m, d) as viewer:
  ee_id = m.site("TCP").id
  target_id = m.body("box").id
  mocap_id  = m.body(target_id).mocapid[0]

  d.mocap_pos[mocap_id] = d.site(ee_id).xpos

  start = time.time()
  circle = Circle(
    center = np.array([0.1, 0.1]), 
    point  = d.mocap_pos[mocap_id],
    freq   = 0.2
  )
  
  joint_names  = [
    "joint1", 
    "joint2", 
    "joint3", 
    "joint4", 
    "joint5", 
    "joint_gripper"
  ]
  dof_ids      = np.array([m.joint   (name).id for name in joint_names])
  actuator_ids = np.array([m.actuator(name).id for name in joint_names])
  body_names   = ['gripper_moving_1', 'gripper_static_1', 'link1_1', 'link2_1', 'link3_1', 'link4_1']
  body_ids     = [m.body(name).id for name in body_names]
  if gravity_compensation: m.body_gravcomp[body_ids] = 1.0

  jac             = np.zeros((6, m.nv))
  diag            = damping * np.eye(6)
  error           = np.zeros(6)
  error_pos       = error[:3]
  error_ori       = error[3:]
  site_quat       = np.zeros(4)
  site_quat_conj  = np.zeros(4)
  error_quat      = np.zeros(4)

  cnt = 0
  while viewer.is_running():
    d.mocap_pos[mocap_id, 0:2] = circle(d.time)

    target = d.mocap_pos[mocap_id].copy()  # move to box
    # target = d.site(ee_id).xpos
    # target[0:2] = circle(d.time)

    error_pos[:] = target - d.site(ee_id).xpos
    # print(error_pos)

    mujoco.mj_jacSite(m, d, jac[:3], None, ee_id)

    dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

    q = d.qpos.copy()
    mujoco.mj_integratePos(m, q, dq, integration_dt)
    qc = np.zeros(q.shape)
    np.clip(q, *m.jnt_range.T, out=qc)
    # print(q[dof_ids] - qc[dof_ids])
    d.ctrl[actuator_ids] = qc[dof_ids]

    robotPlot.next()

    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
    cnt += 1