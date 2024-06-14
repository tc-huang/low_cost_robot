from robot import Robot
from dynamixel import Dynamixel


baudrate=57600
follower_dynamixel = Dynamixel.Config(baudrate=baudrate, device_name='/dev/tty.usbmodem578E0212131').instantiate()
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])

initial_position = [int(4096 / 2)] * 6
print(f"Inital position: {initial_position}")
follower.set_goal_pos(initial_position)

current_position = follower.read_position()
print(f"Current position: {current_position}")