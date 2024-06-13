import cv2 
import numpy as np
import matplotlib.pyplot as plt
from simulation.interface import SimulatedRobot

class RobotPlot:
    def __init__(self, robot: SimulatedRobot, period: float, frames: int, length: float):
        # save config
        self.robot  = robot     # robot
        self.period = period    # timestamp (unit: second)
        self.frames = frames    # the amount of frames between update plot
        self.length = length    # display max length of time (unit: second)

        # create canvas and variables
        self.curr_time    = 0
        self.count        = 0
        self.ani          = None

        # initial content (id, title, is_angle, value_generater, content)
        self.times   = np.zeros((frames), dtype=float)
        self.content = [
            [(0, 0) , "Joint 1 xyz"     , False, lambda: robot.read_ee_pos("link1_1")         , np.zeros((frames, 3), dtype=float)],
            [(1, 0) , "Joint 2 xyz"     , False, lambda: robot.read_ee_pos("link2_1")         , np.zeros((frames, 3), dtype=float)],
            [(2, 0) , "Joint 3 xyz"     , False, lambda: robot.read_ee_pos("link3_1")         , np.zeros((frames, 3), dtype=float)],
            [(3, 0) , "Joint 4 xyz"     , False, lambda: robot.read_ee_pos("link4_1")         , np.zeros((frames, 3), dtype=float)],
            [(4, 0) , "Joint 5 xyz"     , False, lambda: robot.read_ee_pos("gripper_static_1"), np.zeros((frames, 3), dtype=float)],
            [(5, 0) , "Joint ee xyz"    , False, lambda: robot.d.site("TCP").xpos             , np.zeros((frames, 3), dtype=float)],
            [(0, 1) , "Joint 1 angle"   , True , lambda: robot.read_position()[0]             , np.zeros((frames   ), dtype=float)],
            [(1, 1) , "Joint 2 angle"   , True , lambda: robot.read_position()[1]             , np.zeros((frames   ), dtype=float)],
            [(2, 1) , "Joint 3 angle"   , True , lambda: robot.read_position()[2]             , np.zeros((frames   ), dtype=float)],
            [(3, 1) , "Joint 4 angle"   , True , lambda: robot.read_position()[3]             , np.zeros((frames   ), dtype=float)],
            [(4, 1) , "Joint 5 angle"   , True , lambda: robot.read_position()[4]             , np.zeros((frames   ), dtype=float)],
            [(5, 1) , "Joint 6 angle"   , True , lambda: robot.read_position()[5]             , np.zeros((frames   ), dtype=float)],
        ]

        # initial ax
        self.reset()
    
    def reset(self):
        self.curr_time = 0
        self.count = 0
        self.fig, self.ax = plt.subplots(6, 2, sharex='all', figsize=(4, 10))
        self.ax[0, 0].set_xlim([0, self.length])
        for id, title, is_angle, _, _ in self.content:
            self.ax[id].set_title(title)
            if(is_angle): self.ax[id].set_ylim([-np.pi, np.pi])
            else        : self.ax[id].set_ylim([-0.3, 0.3])
        plt.tight_layout()
    
    def plot(self):
        self.count = 0
        if(self.curr_time > self.length):
            self.ax[0, 0].set_xlim([self.curr_time - self.length, self.curr_time])
        for id, _, is_angle, _, content in self.content:
            if(is_angle):
                self.ax[id].plot(self.times, content, 'k')    # theta (radian)
            else:
                self.ax[id].plot(self.times, content[:, 0], 'r') # x
                self.ax[id].plot(self.times, content[:, 1], 'g') # y
                self.ax[id].plot(self.times, content[:, 2], 'b') # z

    def next(self):
        # get data
        self.times[self.count] = self.curr_time
        for _, _, is_angle, f, content in self.content:
            if(is_angle):
                content[self.count] = f()
            else:
                x, y, z = f()
                content[self.count, 0] = x
                content[self.count, 1] = y
                content[self.count, 2] = z
        
        # update count and time
        self.count += 1
        self.curr_time += self.period
        
        # decide whether ploting
        if(self.count == self.frames):
            self.plot()
            self.fig.canvas.draw()
            img_plot = np.array(self.fig.canvas.renderer.buffer_rgba())
            cv2.imshow('Plot', cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            