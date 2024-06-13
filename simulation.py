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
  

# with open('workspace.pkl', 'rb') as f:
#   workspace = pkl.load(f)

# init_pos_idx = np.random.randint(0, len(workspace))
# target_xpos_idx = np.random.randint(0, len(workspace))


# init_pos = workspace[init_pos_idx]['pos']
# target_xpos = workspace[target_xpos_idx]['TCP']

# for i in workspace:
#   print(i['pos'], i['TCP'])
  
# exit()

# print(init_pos_idx, init_pos)
# print(target_xpos_idx, target_xpos)

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
          print(workspace[-1])
          pkl.dump(workspace, f)
          if(len(workspace) == 100):
            exit()
          
      cnt = 0
      pos = random_gen_pos()
    cnt += 1
    r.set_target_pos(pos)
    
    
    # if(cnt < 1000):
    #   r.set_target_pos(init_pos)
    # else:
    #   r.set_target_pos(r.inverse_kinematics(target_xpos))
    # cnt += 1
    # print(cnt,end = '\r')
    
    
    step_start = time.time()
    mujoco.mj_step(m, d)
    viewer.sync()
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)