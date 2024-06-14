import argparse
import pandas as pd
from robot import Robot
from dynamixel import Dynamixel
import time

def rad2step(rad):
    return int((rad / 3.14 + 1) * 2048)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='circle', help='Task to perform')
    args = parser.parse_args()
    task = args.task
    
    baudrate=57600
    follower_dynamixel = Dynamixel.Config(baudrate=baudrate, device_name='/dev/tty.usbmodem578E0212131').instantiate()
    follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])

    initial_position = [rad2step(0.0)] * 6
    print(f"Inital position: {initial_position}")
    follower.set_goal_pos(initial_position)

    current_position = follower.read_position()
    print(f"Current position: {current_position}")
    
    if task == 'line':
        df = pd.read_csv('./line.csv')
    elif task == 'circle':
        df = pd.read_csv('./circle.csv')
    elif task == 'init':
        exit()
    else:
        print('Invalid task')
        exit()
    l = len(df)
    for i in range(l):
        target_position = df.iloc[i][['j0', 'j1', 'j2', 'j3', 'j4', 'j5']].values
        target_position = [rad2step(x) for x in target_position]
        print(f"Target position[{i}/{l}]: {target_position}")
        follower.set_goal_pos(target_position)
        # time.sleep(0.0001)
        current_position = follower.read_position()
        print(f"Current position[{i}/{l}]: {current_position}")
 