import time
import mujoco
import mujoco.viewer

from simulation.interface import SimulatedRobot

import numpy as np
import pickle as pkl

m = mujoco.MjModel.from_xml_path('simulation/low_cost_robot/scene.xml')
d = mujoco.MjData(m)

r = SimulatedRobot(m, d)

def random_gen_pos():
  rst = np.random.uniform(-3.14158 / 2, 3.14158 / 2, 6)
  # rst[2] = np.random.uniform(-3.14158 / 2, 3.14158 / 2)
  rst[4] = 0
  rst[5] = 0
  return rst

cnt = 0
workspace = []
  
with mujoco.viewer.launch_passive(m, d) as viewer:
  start = time.time()
  while viewer.is_running():
    
    
    if(cnt % 5000 == 0):
      if(cnt != 0):
        workspace.append(
          {
            'pos': r.read_position(),
            'TCP': r.read_ee_pos()
          }
        )
      
      with open('workspace.pkl', 'wb') as f:
        pkl.dump(workspace, f)
        if(len(workspace) == 100):
          exit()
        
      cnt = 0
      pos = random_gen_pos()
    r.set_target_pos(pos)
    cnt += 1
    
    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)