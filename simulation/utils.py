import pickle as pkl

def read_pkl(file_name):
    with open(file_name, 'rb') as f:
        data = pkl.load(f)  
    return data

import numpy as np

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
    
    self.start       = point
    self.half_period = period / 2
    self.vec         = (another_point - point) / self.half_period
    self.sign        = True  # Is time change by postive 
    self.prev_t      = 0
    self.temp_t      = 0

  def __call__(self, t: float):
    diff_t = t - self.prev_t
    self.temp_t += diff_t if (self.sign) else -diff_t
    self.prev_t = t
    if (self.temp_t < 0) or (self.temp_t > self.half_period):
      self.temp_t = 2 * (self.half_period if self.sign else 0) - self.temp_t
      self.sign   = not self.sign
    # print(self.temp_t)
    ret = self.start + self.temp_t * self.vec
    # print(ret)
    return ret
