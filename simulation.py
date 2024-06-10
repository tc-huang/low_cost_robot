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


def mse(x, y):
  return np.mean((np.array(x) - np.array(y)) ** 2)

with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  
  
  [0.817, -1.1, 0.723, -0.0314, -0.775]
  ee_target = [-0.16894777, 0.17502224, 0.05318767]
  
  last = [0, 0, 0, 0, 0]
  current = [0, 0, 0, 0, 0]
  
  is_picked = False
  is_open = True
  cnt = 0
  while viewer.is_running():
    

    # print('joint1', r.read_ee_pos('joint1'))
    # print('joint2', r.read_ee_pos('joint2'))
    # print('joint3', r.read_ee_pos('joint3'))
    # print('joint4', r.read_ee_pos('joint4'))
    # print('joint4-pad', r.read_ee_pos('joint4-pad'))
    # print('joint5', r.read_ee_pos('joint5'))
    # print('joint5-pad', r.read_ee_pos('joint5-pad'))
    # print('box', r.read_ee_pos('box'))
    # print(r.read_ee_pos())
    if(is_picked):
      target_pos = r.inverse_kinematics(r.read_ee_pos('box'))
      target_pos[1] = target_pos[1] - target_pos[1] * 0.1
    else:
      target_pos = r.inverse_kinematics(r.read_ee_pos('box'))
    
    
    if(is_open):
      target_pos[5] = -0.7
    else:
      target_pos[5] = -0.3
    
    
    if(cnt == 700):
      is_picked = True
      is_open = False
    
    cnt += 1
    print('1', r.read_ee_pos())
    print('2', r.read_ee_pos('box'))
        
    time.sleep(0.05)
    
    r.set_target_pos(target_pos)
    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)