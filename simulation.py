import time
import mujoco
import mujoco.viewer

from simulation.interface import SimulatedRobot

import numpy as np
import pickle as pkl

class SimulatedRobotClass:

  def __init__(self) -> None:
    self.m = mujoco.MjModel.from_xml_path('simulation/low_cost_robot/scene.xml')
    self.d = mujoco.MjData(self.m)
    self.r = SimulatedRobot(self.m, self.d)
    self.cnt = 0
    self.workspace = []
    self.workspace = self.read_pkl('workspace.pkl')
    self.init_pos_idx, self.target_xpos_idx, self.init_pos, self.target_xpos = self.get_simulation_position_xpos()

  def random_gen_pos(self):
    rst = np.random.uniform(-3.14158 / 2, 3.14158 / 2, 6)
    # rst[2] = np.random.uniform(-3.14158 / 2, 3.14158 / 2)
    rst[4] = 0
    rst[5] = 0
    return rst
  
  def random_gen_position_xpos(self):
    if(self.cnt % 2000 == 0):
      if(self.cnt != 0):
        self.workspace.append(
          {
            'pos': self.r.read_position().copy(),
            'TCP': self.r.read_ee_pos().copy()
          }
        )
      
        with open('workspace.pkl', 'wb') as f:
          print(self.workspace[-1])
          pkl.dump(self.workspace, f)
          if(len(self.workspace) == 100):
            exit()
          
      self.cnt = 0
      self.pos = self.random_gen_pos()
    self.cnt += 1
    print(self.cnt, end = '\r')
    self.r.set_target_pos(self.pos)

  def read_pkl(self, file_names):
    with open(file_names, 'rb') as f:
      workspace = pkl.load(f)  
    return workspace

  def get_simulation_position_xpos(self):
    init_pos_idx = np.random.randint(0, len(self.workspace))
    target_xpos_idx = np.random.randint(0, len(self.workspace))
    init_pos = self.workspace[init_pos_idx]['pos']
    target_xpos = self.workspace[target_xpos_idx]['TCP']

    for i in self.workspace:
      print(i['pos'], i['TCP'])
    # exit()

    print(init_pos_idx, init_pos)
    print(target_xpos_idx, target_xpos)

    return init_pos_idx, target_xpos_idx, init_pos, target_xpos

  def run(self):
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      start = time.time()
      while viewer.is_running():
        
        
        # self.random_gen_position_xpos()
        
        if(self.cnt < 200):
          self.r.set_target_pos(self.init_pos)
        else:
          for i in self.r.inverse_kinematics(self.target_xpos):
            self.r.set_target_pos(i)
            self.cnt += 1
            print(self.cnt, np.subtract(self.r.read_ee_pos(), self.target_xpos),end = '\r')
            step_start = time.time()
            mujoco.mj_step(self.m, self.d)
            viewer.sync()
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
              time.sleep(time_until_next_step)



        self.cnt += 1
        print(self.cnt, np.subtract(self.r.read_ee_pos(), self.target_xpos),end = '\r')
        step_start = time.time()
        mujoco.mj_step(self.m, self.d)
        viewer.sync()
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)

if __name__ == '__main__':
  r = SimulatedRobotClass()
  r.run()