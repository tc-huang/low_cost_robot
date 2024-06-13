import cv2
import time
import mujoco
import numpy as np
import pickle as pkl
import mujoco.viewer
from simulation.plot import RobotPlot
from simulation.interface import SimulatedRobot
from simulation.utils import read_pkl

np.random.seed(42)

class Simulation:

  def __init__(self, n_epoch=100) -> None:
    self.m = mujoco.MjModel.from_xml_path('simulation/low_cost_robot/scene.xml')
    self.d = mujoco.MjData(self.m)
    self.r = SimulatedRobot(self.m, self.d)
    self.robotPlot = RobotPlot(self.r, period=self.m.opt.timestep, frames=60, length=0.3)
    self.cnt = 0
    self.workspace = read_pkl('workspace.pkl')
    self.n_workspace = len(self.workspace)
    self.n_epoch = n_epoch

  def random_gen_pose(self)->np.ndarray:
    """Generate a random angle for the 6 joints of the robot.

    Returns:
        np.ndarray: 6 random angles for the 6 joints of the robot. (in radians)
    """
    joint_cfg = np.random.uniform(-3.14158 / 2, 3.14158 / 2, 6)
    joint_cfg[4] = 0
    joint_cfg[5] = 0
    return joint_cfg
  
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
    self.r.set_target_pose(self.pos)

  def get_diff_idxs(self):
    init_pos_idx = np.random.randint(0, self.n_workspace-1)
    diff = np.random.randint(1, self.n_workspace-1)
    target_xpos_idx = (init_pos_idx + diff) % self.n_workspace
    assert(init_pos_idx != target_xpos_idx)
    return init_pos_idx, target_xpos_idx

  def get_simulation_position_xpos(self):
    init_pos_idx, target_xpos_idx = self.get_diff_idxs()
    init_pos = self.workspace[init_pos_idx]['pos']
    target_xpos = self.workspace[target_xpos_idx]['TCP']
    return init_pos_idx, target_xpos_idx, init_pos, target_xpos
  
  def next_pos(self):
    while self.cnt < 200:
      yield self.init_pos
    for next_pos, error in self.r.inverse_kinematics(self.target_xpos):
      print(f"\r{self.cnt}: error: {error} norm = {np.linalg.norm(error):.10f}              ", end='')
      self.robotPlot.next() # plot
      yield next_pos
    print()
    return

  def run(self):
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # start = time.time()
      is_running = viewer.is_running()
      epoch = 0
      while is_running and (epoch < self.n_epoch):
        print(f"epoch: {epoch}")
        self.cnt = 0
        self.init_pos_idx, self.target_xpos_idx, self.init_pos, self.target_xpos = self.get_simulation_position_xpos()
        self.robotPlot.reset()
        for i in self.next_pos():
          self.r.set_target_pos(i)
          self.cnt += 1
          # print(self.cnt, np.subtract(self.r.read_ee_pos(), self.target_xpos),end = '\r')
          step_start = time.time()
          mujoco.mj_step(self.m, self.d)
          viewer.sync()
          # Rudimentary time keeping, will drift relative to wall clock.
          time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
          if time_until_next_step > 0:
            time.sleep(time_until_next_step)
          is_running = viewer.is_running()
          if not is_running: break
        cv2.waitKey()
        epoch += 1

if __name__ == '__main__':
  r = Simulation(100)
  r.run()