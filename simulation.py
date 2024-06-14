import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import time
import mujoco
import argparse
import numpy as np
import pickle as pkl
import mujoco.viewer
from simulation.plot import RobotPlot
from simulation.interface import SimulatedRobot
from simulation.utils import read_pkl

from ik.gradient_descent import GradientDescentIK
from ik.gauss_newton import GaussNewtonIK
from ik.levenberg_marquardt import LevenbegMarquardtIK

np.random.seed(42)

class Circle:
  def __init__(self, point: np.ndarray, other: dict):
    center = other['center']
    if not isinstance(center, np.ndarray):
      center = np.array(center)
    assert(point .shape == (2, ))
    assert(center.shape == (2, ))
    freq   = other['freq']
    diff = point - center
    x, y = diff

    self.center = center
    self.theta  = np.arctan2([y], [x])[0]  # compute the angle
    self.omega  = 2 * np.pi * freq
    self.radius = np.linalg.norm(diff)
    print(self.center, self.theta, self.omega, self.radius)

  def __call__(self, t: float):
    angle = self.omega * t + self.theta
    ret = self.radius * np.array([np.cos(angle), np.sin(angle)]) + self.center
    # print(ret)
    return ret
  
class Line:
  def __init__(self, point: np.ndarray, other: dict):
    another_point = other['another_point']
    period        = other['period']
    if(not isinstance(another_point, np.ndarray)):
      another_point = np.array(another_point)
    
    assert(point        .shape == (2, ))
    assert(another_point.shape == (2, ))
    
    self.period = period
    self.start  = point
    self.vec    = (another_point - point) / period
    self.sign   = True  # Is time change by postive 
    self.prev_t = 0
    self.temp_t = 0

  def __call__(self, t: float):
    diff_t = t - self.prev_t
    self.temp_t += diff_t if (self.sign) else -diff_t
    self.prev_t = t
    if (self.temp_t < 0) or (self.temp_t > self.period):
      self.temp_t = 2 * (self.period if self.sign else 0) - self.temp_t
      self.sign   = not self.sign
    print(self.temp_t)
    ret = self.start + self.temp_t * self.vec
    print(ret)
    return ret

class Simulation:

  def __init__(self, n_epoch=100) -> None:
    self.m = mujoco.MjModel.from_xml_path('simulation/low_cost_robot/scene.xml')
    self.d = mujoco.MjData(self.m)
    self.r = SimulatedRobot(self.m, self.d)
    self.robotPlot = RobotPlot(self.r, period=self.m.opt.timestep, frames=60, length=2)
    self.cnt = 0
    self.time = 0.0
    self.workspace = read_pkl('workspace.pkl')
    self.n_workspace = len(self.workspace)
    self.n_epoch = n_epoch
    self.trace = None

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
    init_xpos = self.workspace[init_pos_idx]['TCP']
    target_pos = self.workspace[target_xpos_idx]['pos']
    target_xpos = self.workspace[target_xpos_idx]['TCP']
    return init_pos_idx, target_xpos_idx, init_pos, init_xpos, target_pos, target_xpos
  
  def next_pos(self):
    # while self.cnt < 200:
    #   yield self.init_pos
    # for next_pos, error in self.r.inverse_kinematics(self.target_xpos):
    #   print(f"\r{self.cnt}: error: {error} norm = {np.linalg.norm(error):.10f}              ", end='')
    #   self.robotPlot.next() # plot
    #   yield next_pos
    # print()
    target_xpos = self.r.read_ee_pos().copy()
    a, b = self.trace(self.time)
    target_xpos[0] = a
    target_xpos[2] = b
    
    for next_pos, error in self.r.inverse_kinematics(target_xpos):
      print(f"\r{self.cnt}: error: {error} norm = {np.linalg.norm(error):.10f}", end='')
      self.robotPlot.next() # plot
      yield next_pos
    print()
    return

  def run(self):
    with mujoco.viewer.launch_passive(model=self.m, data=self.d, show_left_ui=False, show_right_ui=False) as viewer:
      # start = time.time()
      x, y, z = self.r.read_ee_pos()
      xpos = np.array([x, z])
      self.trace = Circle(xpos, other={"center": [0.1, 0.1], "freq": 0.5})
      # self.trace = Line(xpos, other={"another_point": [0.2, 0.2], "period": 2})
      is_running = viewer.is_running()
      epoch = 0
      data = []
      # while is_running: # and (epoch < self.n_epoch):
      while is_running and (epoch < self.n_epoch):
        print(f"epoch: {epoch}")
        self.cnt = 0
        self.time += self.m.opt.timestep
        # self.init_pos_idx, self.target_xpos_idx, self.init_pos, self.target_xpos = self.get_simulation_position_xpos()
        # self.robotPlot.reset()
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
          # print(self.m.opt.timestep)
          self.time += self.m.opt.timestep
          # print(">", self.time)
          is_running = viewer.is_running()
          if not is_running: break
        x, y, z = self.r.read_ee_pos()
        j0, j1, j2, j3, j4, j5 = self.r.read_position()
        data.append({"x": x, "y": y, "z": z, "j0": j0, "j1": j1, "j2": j2, "j3": j3, "j4": j4, "j5": j5})
          
        # if(cv2.waitKey(10) == ord('q')): break
        epoch += 1
      df = pd.DataFrame(data)
      with open("circle.csv", "wt") as f:
        df.to_csv(f)
  
  def plot_exp1_samples(self, target_path):
    df = pd.read_csv(f'{target_path}.csv')
    print(df)
    plt.figure('plot_samples')
    plt.scatter(df['xpos_error'], df['iterations'])
    plt.xlabel('xpos_error')
    plt.ylabel('iterations')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(f'{target_path}.png')

  def exp1(self, ik_method,samples = 100, tol = 0.001, limit_iter = 200, file_name = 'exp1'):
    #Init variables.
    model = self.m
    data = self.d
    jacp = np.zeros((3, model.nv)) #translation jacobian
    jacr = np.zeros((3, model.nv)) #rotational jacobian
    step_size = 0.5
    alpha = 0.5
    damping = 0.15
    

    ik_method_dict = {
      'GradientDescent': GradientDescentIK,
      'GaussNewton': GaussNewtonIK,
      'LevenbergMarquardt': LevenbegMarquardtIK
    }
    
    if ik_method not in ik_method_dict:
      print("Invalid ik_method")
      return
    
    if not os.path.exists(f'./exp_data/{ik_method}'):
      os.makedirs(f'./exp_data/{ik_method}')

    results = []

    for i in range(samples):
      init_pos_idx, target_xpos_idx, init_pos, init_xpos, target_pos, target_xpos = self.get_simulation_position_xpos()
      configs = {
        'model':model,
        'data':data,
        'step_size':step_size,
        'tol':tol,
        'alpha':alpha,
        'jacp':jacp,
        'jacr':jacr,
        'damping':damping
      }
      ik = ik_method_dict[ik_method](**configs)
      #Get desire point
      mujoco.mj_resetDataKeyframe(model, data, 1) #reset qpos to initial value
      xpos_error, iterations = ik.calculate(target_xpos, init_pos, limit_iter) #calculate the qpos
      print(i, xpos_error, iterations, end = '\r')

      results.append(
        {
          'xpos_error':xpos_error,
          'iterations':iterations,
          'init_pos_idx':init_pos_idx,
          'target_xpos_idx':target_xpos_idx,
          'init_pos':init_pos, 
          'init_xpos':init_xpos,
          'target_pos':target_pos,
          'target_xpos':target_xpos
        }
      )

    df = pd.DataFrame(results)
    df.to_csv(f'./exp_data/{ik_method}/{file_name}.csv', index = False)

    self.plot_exp1_samples(f'./exp_data/{ik_method}/{file_name}')
        
    
    
    
    
    
    
    


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--n_epoch', type=int, default=100, help='Number of epochs')
  parser.add_argument('--task', type=str, default='plot_samples', help='Task to perform')
  parser.add_argument('--ik_method', type=str, default='GaussNewton', help='IK method to use GradientDescent, GaussNewton, LevenbergMarquardt')



  args = parser.parse_args()
  n_epoch = args.n_epoch
  task = args.task
  
  r = Simulation(n_epoch)
  if task == 'run':
    r.run()
  if task == 'exp1':
    r.exp1(args.ik_method, samples = 1000,tol = 1e-3, limit_iter = 2e4, file_name='tol_0_001')
  if task == 'plot_samples':
    r.plot_exp1_samples(target_path = './exp_data/GaussNewton/tol_0_001')