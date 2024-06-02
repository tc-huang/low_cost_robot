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


# Mocap body we will control with our mouse.

def check_joint_limits(q):
    """Check if the joints is under or above its limits"""
    for i in range(len(q)):
        q[i] = max(m.jnt_range[i][0], min(q[i], m.jnt_range[i][1]))
Kpos: float = 0.95
mocap_name = "box"
mocap_id = m.body(mocap_name).mocapid[0]
ee_name = "end_effector"

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    robotPlot.next()


    mocap_coor = d.mocap_pos[mocap_id]
    ee_coor    = r.read_ee_pos(ee_name)
    print('\033[H\033[2J', mocap_coor, ee_coor, sep='\n')
    dx = mocap_coor - ee_coor
    target_pos = ee_coor + Kpos * dx
    mid = r.inverse_kinematics(target_pos, ee_name)
    check_joint_limits(mid)
    r.set_target_pos(mid)


    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)