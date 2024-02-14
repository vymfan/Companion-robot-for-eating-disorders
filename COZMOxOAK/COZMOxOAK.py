from depthai_sdk import OakCamera, TextPosition, Visualizer
from depthai_sdk.classes.packets  import TwoStagePacket
import numpy as np
import cv2
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

def rotate_cozmo(robot: cozmo.robot.Robot):
    # Tilt Cozmo's head 30 degrees downwards
    robot.move_head(degrees(-30)).wait_for_completed()

    # Pause for 3 seconds
    time.sleep(3)

    # Change direction: make Cozmo turn 90 degrees to the left
    robot.turn_in_place(degrees(90)).wait_for_completed()

    # Move Cozmo forward for 100 millimeters at a speed of 50 millimeters per second for 1 second
    robot.drive_straight(distance_mm(100), speed_mmps(50)).wait_for_completed()

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

with OakCamera() as oak:
    color = oak.create_camera('color')
    det_nn = oak.create_nn('face-detection-retail-0004', color)
    # Passthrough is enabled for debugging purposes
    # AspectRatioResizeMode has to be CROP for 2-stage pipelines at the moment
    det_nn.config_nn(resize_mode='crop')

    emotion_nn = oak.create_nn('emotions-recognition-retail-0003', input=det_nn)
    # emotion_nn.config_multistage_nn(show_cropped_frames=True) # For debugging

    def cb(packet: TwoStagePacket):
        vis: Visualizer = packet.visualizer
        for det, rec in zip(packet.detections, packet.nnData):
            emotion_results = np.array(rec.getFirstLayerFp16())
            emotion_name = emotions[np.argmax(emotion_results)]

            # Afficher le texte dans la sortie de VSCode
            print(f"Émotion détectée : {emotion_name}")

            vis.add_text(emotion_name,
                            bbox=(*det.top_left, *det.bottom_right),
                            position=TextPosition.BOTTOM_RIGHT)
            
            if emotion_name == 'sad':
                # Run the program defined by the rotate_cozmo function
                cozmo.run_program(rotate_cozmo)

        vis.draw(packet.frame)
        cv2.imshow(packet.name, packet.frame)


    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(emotion_nn, callback=cb, fps=True)
    oak.visualize(det_nn.out.passthrough)
    # oak.show_graph() # Show pipeline graph
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)
