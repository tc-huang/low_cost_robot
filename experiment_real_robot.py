from robot import Robot
from dynamixel import Dynamixel

def rad2step(rad):
    return int((rad / 3.14 + 1) * 2048)


baudrate=57600
follower_dynamixel = Dynamixel.Config(baudrate=baudrate, device_name='/dev/tty.usbmodem578E0212131').instantiate()
follower = Robot(follower_dynamixel, servo_ids=[1, 2, 3, 4, 5, 6])

initial_position = [rad2step(0.0)] * 6
print(f"Inital position: {initial_position}")
follower.set_goal_pos(initial_position)

current_position = follower.read_position()
print(f"Current position: {current_position}")