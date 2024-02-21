from depthai_sdk import OakCamera, TextPosition, Visualizer
from depthai_sdk.classes.packets  import TwoStagePacket
import numpy as np
import cv2
import cozmo
from cozmo.util import degrees, distance_mm, speed_mmps

emotions = ['neutral', 'happy', 'sad', 'surprise', 'anger']

def cozmo_program(robot: cozmo.robot.Robot):
    # Drive forwards for 150 millimeters at 50 millimeters-per-second.
    robot.drive_straight(distance_mm(150), speed_mmps(50)).wait_for_completed()

    # Turn 90 degrees to the left.
    # Note: To turn to the right, just use a negative number.
    robot.turn_in_place(degrees(90)).wait_for_completed()

    # Tell the head motor to start lowering the head (at 5 radians per second)
    robot.move_head(-5)
    # Tell the lift motor to start lowering the lift (at 5 radians per second)
    robot.move_lift(-5)
    # Turn 90 degrees to the left.
    # Note: To turn to the right, just use a negative number.
    robot.turn_in_place(degrees(45)).wait_for_completed()

    # wait for 1 seconds (the head, lift and wheels will move while we wait)
    time.sleep(1)

    # Tell the head motor to start raising the head (at 5 radians per second)
    robot.move_head(5)
    # Tell the lift motor to start raising the lift (at 5 radians per second)
    robot.move_lift(5)
    # Turn 90 degrees to the right.
    robot.turn_in_place(degrees(-45)).wait_for_completed()

    # wait for 1 seconds (the head, lift and wheels will move while we wait)
    time.sleep(1)

    # Tell the head motor to start lowering the head (at 5 radians per second)
    robot.move_head(-5)
    # Tell the lift motor to start lowering the lift (at 5 radians per second)
    robot.move_lift(-5)
    # Turn 90 degrees to the left.
    # Note: To turn to the right, just use a negative number.
    robot.turn_in_place(degrees(45)).wait_for_completed()

    # wait for 1 seconds (the head, lift and wheels will move while we wait)
    time.sleep(1)

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

            if emotion_name == 'surprise':
                cozmo.run_program(cozmo_program)

            vis.add_text(emotion_name,
                            bbox=(*det.top_left, *det.bottom_right),
                            position=TextPosition.BOTTOM_RIGHT)

        vis.draw(packet.frame)
        cv2.imshow(packet.name, packet.frame)


    # Visualize detections on the frame. Also display FPS on the frame. Don't show the frame but send the packet
    # to the callback function (where it will be displayed)
    oak.visualize(emotion_nn, callback=cb, fps=True)
    oak.visualize(det_nn.out.passthrough)
    # oak.show_graph() # Show pipeline graph
    oak.start(blocking=True) # This call will block until the app is stopped (by pressing 'Q' button)