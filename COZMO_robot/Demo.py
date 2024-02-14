import cozmo
import time
import pycozmo
from cozmo.util import degrees

# Establish a connection to Cozmo using the pycozmo library
with pycozmo.connect() as cli:
    # Set Cozmo's head angle to 0.7 radians
    cli.set_head_angle(angle=0.7)
    
    # Pause for 1 second to allow Cozmo to adjust its head angle
    time.sleep(1)

    # Drive Cozmo forward with left and right wheel speeds of 50.0 for a duration of 4 seconds
    cli.drive_wheels(lwheel_speed=50.0, rwheel_speed=50.0, duration=4.0)

# Pause to ensure a stable connection with pycozmo before proceeding to the next command
time.sleep(1)

# Initialize a new pycozmo client
cli = pycozmo.Client()

# Start the client
cli.start()

# Connect the client to Cozmo
cli.connect()

# Wait for the robot to be connected and ready
cli.wait_for_robot()

# Change direction: make Cozmo turn 90 degrees to the right
cli.drive_wheels(lwheel_speed=50.0, rwheel_speed=-50.0, duration=2.0)

# Disconnect from Cozmo
cli.disconnect()

# Stop the client
cli.stop()

def move_head(robot: cozmo.robot.Robot):
    # Tilt Cozmo's head 30 degrees downwards
    robot.move_head(degrees(-30)).wait_for_completed()

    # Pause for 3 seconds
    time.sleep(3)

    # Change direction: make Cozmo turn 90 degrees to the left
    robot.turn_in_place(degrees(90)).wait_for_completed()

    # Move Cozmo forward for 100 millimeters at a speed of 50 millimeters per second for 1 second
    robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()

# Run the program defined by the move_head function
cozmo.run_program(move_head)