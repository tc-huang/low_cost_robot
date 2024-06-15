from robot import Robot
from dynamixel import Dynamixel

baudrate=57600
leader_dynamixel = Dynamixel.Config(baudrate=baudrate, device_name='/dev/tty.usbmodem578E0213781').instantiate()
leader = Robot(leader_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])
leader.set_trigger_torque()
follower_dynamixel = Dynamixel.Config(baudrate=baudrate, device_name='/dev/tty.usbmodem578E0212131').instantiate()
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])


while True:
    follower.set_goal_pos(leader.read_position())