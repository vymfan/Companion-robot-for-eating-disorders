import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu  # Import a custom module 'mediapipe_utils' and alias it as 'mpu'

import depthai as dai  # Import the 'depthai' library for working with the DepthAI hardware
import cv2  # Import the OpenCV library for computer vision
from pathlib import Path  # Import the 'Path' class from the 'pathlib' module for working with file paths
from FPS import FPS, now  # Import custom modules for frame rate measurement

import time  # Import the 'time' module for working with time-related functions
import sys  # Import the 'sys' module for interacting with the Python interpreter
from string import Template  # Import the 'Template' class from the 'string' module for string templating
import marshal  # Import the 'marshal' module for object serialization

from HostSpatialCalc import HostSpatialCalc  # Import a custom module 'HostSpatialCalc'
from face_geometry import (  # Import specific functions/classes from the 'face_geometry' module
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
    canonical_metric_landmarks
)
from scipy.spatial import distance as dist  # Import the 'distance' function from the 'scipy.spatial' module

# Define the path to the directory of the current script
SCRIPT_DIR = Path(__file__).resolve().parent

# Define paths to various model files using 'SCRIPT_DIR'
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_pp_top2_th50_sh4.blob")
HAND_LANDMARK_MODEL = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
HAND_TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "hand_template_manager_script_solo.py")
HAND_TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "hand_template_manager_script_duo.py")
FACE_DETECTION_MODEL = str(SCRIPT_DIR / "models/face_detection_short_range_pp_top1_th50_sh1.blob")
FACE_LANDMARK_MODEL = str(SCRIPT_DIR / "models/face_landmark_pp_sh4.blob")
FACE_LANDMARK_WITH_ATTENTION_MODEL = str(SCRIPT_DIR / "models/face_landmark_with_attention_pp_sh4.blob")
FACE_TEMPLATE_MANAGER_SCRIPT = str(SCRIPT_DIR / "face_template_manager_script.py")

class DepthSync:
    """
    Class to manage synchronization of depth frames with RGB frames.
    Stores history of depth frames to ensure alignment with RGB frames.
    """
    def __init__(self):
        # Initialize an empty list to store depth frames
        self.msg_lst = []

    def add(self, msg):
        """
        Add a message or a list of messages to the history list.
        """
        # Check if 'msg' is a list, and if so, append each element to the history list
        if isinstance(msg, list):
            for m in msg:
                self.msg_lst.append(m)
        else:
            # If 'msg' is not a list, append it as a single element to the history list
            self.msg_lst.append(msg)

    def get(self, msg):
        """
        Return the message from the list with the closest timestamp to 'msg' timestamp,
        and remove older messages from the list.
        """
        # Check if the history list is empty, and if so, return None
        if len(self.msg_lst) == 0:
            return None
        # Get the timestamp of the input message 'msg'
        ts = msg.getTimestamp()
        # Calculate the time difference between 'msg' and the first message in the history list
        delta_min = abs(ts - self.msg_lst[0].getTimestamp())
        msg_id = 0
        # Iterate through the remaining messages in the history list
        for i in range(1, len(self.msg_lst)):
            # Calculate the time difference between 'msg' and the current message
            delta = abs(ts - self.msg_lst[i].getTimestamp())
            # Check if the current message has a closer timestamp to 'msg' than the previous minimum
            if delta < delta_min:
                delta_min = delta
                msg_id = i
            else:
                break
        # Retrieve the message with the closest timestamp from the history list
        msg = self.msg_lst[msg_id]
        # Remove older messages from the history list
        del self.msg_lst[:msg_id + 1]
        # Return the retrieved message
        return msg


class HandFaceTracker:
    """
    Mediapipe Hand and Face Tracker for depthai (= Mediapipe Hand tracker + Mediapipe Facemesh)
    Arguments:
    - input_src: frame source,
            - "rgb" or None: OAK* internal color camera,
            - a file path of an image or a video,
            - an integer (e.g., 0) for a webcam id,
    - nb_hands: 0, 1, or 2. Number of hands max tracked. If 0, then hand tracking is not used. 1 is faster than 2.
    - use_face_pose: boolean. If yes, compute the face pose transformation matrix and the metric landmarks.
            The face pose transformation matrix provides mapping from the static canonical face model to the runtime face.
            The metric landmarks are the 3D runtime metric landmarks aligned with the canonical metric face landmarks (unit: cm).
    - xyz: boolean, when True calculate the (x, y, z) coords of face (measure on the forehead) and hands.
    - crop: boolean which indicates if square cropping on source images is applied or not
    - internal_fps: when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution: sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height: when using the internal color camera, set the frame height (calling setIspScale()).
            The width is calculated accordingly to height and depends on the value of 'crop'
    - use_gesture: boolean, when True, recognize hand poses from a predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - single_hand_tolerance_thresh (when nb_hands=2 only): if there is only one hand in a frame,
            in order to know when a second hand will appear you need to run the palm detection
            in the following frames. Because palm detection is slow, you may want to delay
            the next time you will run it. 'single_hand_tolerance_thresh' is the number of
            frames during only one hand is detected before palm detection is run again.
    - focus: None or int between 0 and 255. Color camera focus.
            If None, auto-focus is active. Otherwise, the focus is set to 'focus'
    - trace: int, 0 = no trace, otherwise print some debug messages or show the output of ImageManip nodes
            if trace & 1, print application-level info like the number of palm detections,
            if trace & 2, print lower-level info like when a message is sent or received by the manager script node,
            if trace & 4, show in cv2 windows outputs of ImageManip node,
            if trace & 8, save in the file tmp_code.py the python code of the manager script node
            Ex: if trace == 3, both application and low-level info are displayed.
    """
    def __init__(self, input_src=None,
                with_attention=True,
                double_face=False,
                nb_hands=2,
                use_face_pose=False,
                xyz=False,
                crop=False,
                internal_fps=None,
                resolution="full",
                internal_frame_height=640,
                use_gesture=False,
                hlm_score_thresh=0.8,
                single_hand_tolerance_thresh=3,
                focus=None,
                trace=0
                ):

        self.pd_model = PALM_DETECTION_MODEL  # Set the palm detection model path
        print(f"Palm detection blob     : {self.pd_model}")

        self.hlm_model = HAND_LANDMARK_MODEL  # Set the hand landmark model path
        self.hlm_score_thresh = hlm_score_thresh  # Set the hand landmark score threshold
        print(f"Landmark blob           : {self.hlm_model}")

        self.fd_model = FACE_DETECTION_MODEL  # Set the face detection model path
        print(f"Face detection blob     : {self.fd_model}")

        self.with_attention = with_attention  # Set whether to use face landmark with attention
        if self.with_attention:
            self.flm_model = FACE_LANDMARK_WITH_ATTENTION_MODEL
        else:
            self.flm_model = FACE_LANDMARK_MODEL
        self.flm_score_thresh = 0.5  # Set the face landmark score threshold
        print(f"Face landmark blob      : {self.flm_model}")

        self.nb_hands = nb_hands  # Set the maximum number of hands to track

        self.xyz = False  # Set whether to calculate (x, y, z) coordinates of face and hands
        self.crop = crop  # Set whether square cropping on source images is applied
        self.use_world_landmarks = True  # Set whether to use world landmarks

        if focus is None:
            self.focus = None  # Set color camera focus to None (auto-focus)
        else:
            self.focus = max(min(255, int(focus)), 0)  # Set color camera focus within valid range

        self.trace = trace  # Set the trace level for debugging
        self.use_gesture = use_gesture  # Set whether to recognize hand poses from predefined set of poses
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh  # Set tolerance for single hand detection delay
        self.double_face = double_face  # Set whether to enable experimental feature for improving FPS

        if self.double_face:
            print("This is an experimental feature that should help to improve the FPS")
            if self.nb_hands > 0:
                print("With double_face flag, the hand tracking is disabled !")
                self.nb_hands = 0  # Disable hand tracking when double_face flag is enabled

        self.device = dai.Device()  # Initialize the DepthAI device
        
        if input_src == None or input_src == "rgb":
            # OAK* internal color camera configuration
            self.input_type = "rgb" # Set the input type to "rgb"
            # Determine sensor resolution based on the provided 'resolution' parameter
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            if xyz:
                # Check if 'xyz' is True and the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras: 
                    self.xyz = True # Stereo is supported, so set 'xyz' to True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            # Set default internal FPS if 'internal_fps' is not provided
            if internal_fps is None:
                if self.double_face:
                    # If 'double_face' is True, set internal FPS based on 'with_attention'
                    if self.with_attention:
                        self.internal_fps = 14
                    else:
                        self.internal_fps = 41
                # If 'double_face' is False, set internal FPS based on 'with_attention' and 'nb_hands'
                else:
                    if self.with_attention:
                        self.internal_fps = 11
                    else:
                        if self.nb_hands == 0:
                            self.internal_fps = 27
                        elif self.nb_hands == 1:
                            self.internal_fps = 24
                        else: # nb_hands = 2
                            self.internal_fps = 19


            # If 'internal_fps' is provided, use the specified value
            else:
                self.internal_fps = internal_fps 
            
            
                # Additional adjustment for internal FPS when using OAK* internal color camera
                if self.input_type == "rgb" and internal_fps is None:
                    if self.with_attention:
                        self.internal_fps = 14
                    else:
                        self.internal_fps = 41
                

            print(f"Internal camera FPS set to: {self.internal_fps}") 

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            # Check if cropping is enabled
            if self.crop:
                # Calculate ISP scale parameters and set image size and padding for cropping
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                # Calculate ISP scale parameters and set image size, padding, and cropping width for non-cropped case
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0
        
            # Print information about the internal camera image size and padding
            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        # Check if the input source is an image file (ends with '.jpg' or '.png')
        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            # Read the image file and set image dimensions
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        
        else:
            # Check if the input source is a video file
            self.input_type = "video"
            # Convert input source to integer if it represents a webcam ID
            if input_src.isdigit():
                input_src = int(input_src)
            # Open the video capture object and retrieve video properties
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)
        
        # If the input type is not "rgb," configure dimensions and cropping for non-rgb input
        if self.input_type != "rgb":
            self.xyz = False
            print(f"Original frame size: {self.img_w}x{self.img_h}")

            # Calculate frame size and cropping/padding dimensions based on cropping configuration
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h)
            else:
                self.frame_size = max(self.img_w, self.img_h)
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)
                     
            # Adjust image dimensions for cropping configuration
            if self.crop: self.img_h = self.img_w = self.frame_size
            print(f"Frame working size: {self.img_w}x{self.img_h}")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues based on the input type and number of hands
        if self.input_type == "rgb":
            # For RGB input, get the video output queue
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=2, blocking=True)
        else:
            # For non-RGB input, get the face manager input queue
            self.q_face_manager_in = self.device.getInputQueue(name="face_manager_in")
        if self.nb_hands > 0:
            # If hand tracking is enabled, get the hand manager output queue
            self.q_hand_manager_out = self.device.getOutputQueue(name="hand_manager_out", maxSize=2, blocking=True)
        # Get queues for face and hand managers, and face landmark neural network
        self.q_face_manager_out = self.device.getOutputQueue(name="face_manager_out", maxSize=2, blocking=True)
        self.q_flm_nn_out = self.device.getOutputQueue(name="flm_nn_out", maxSize=2, blocking=True)
        # For debugging, get queues for ImageManip node outputs
        if self.trace & 4:
            if self.nb_hands > 0:
                self.q_pre_pd_manip_out = self.device.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
                self.q_pre_hlm_manip_out = self.device.getOutputQueue(name="pre_hlm_manip_out", maxSize=1, blocking=False)    
            self.q_pre_fd_manip_out = self.device.getOutputQueue(name="pre_fd_manip_out", maxSize=1, blocking=False)
            self.q_pre_flm_manip_out = self.device.getOutputQueue(name="pre_flm_manip_out", maxSize=1, blocking=False)    
        # If depth calculation (xyz) is enabled, get queues for depth, initialize DepthSync, and create HostSpatialCalc
        if self.xyz:
            self.q_depth_out = self.device.getOutputQueue(name="depth_out", maxSize=5, blocking=True)
            self.depth_sync = DepthSync()
            self.spatial_calc = HostSpatialCalc(self.device, delta=int(self.img_w/100), thresh_high=3000)
       
        # Initialize FPS and sequence number
        self.fps = FPS()
        self.seq_num = 0

        # Check if face pose computation is enabled
        self.use_face_pose = use_face_pose
        if self.use_face_pose:
            # Read calibration data and get RGB camera intrinsics and distortion coefficients
            calib_data = self.device.readCalibration()
            self.rgb_matrix= np.array(calib_data.getCameraIntrinsics(dai.CameraBoardSocket.RGB, resizeWidth=self.img_w, resizeHeight=self.img_h))
            self.rgb_dist_coef = np.array(calib_data.getDistortionCoefficients(dai.CameraBoardSocket.RGB))
            # Create a PCF (Perspective Camera Frustum) object for face pose computation
            self.pcf = PCF(
                near=1,
                far=10000,
                frame_height=self.img_h,
                frame_width=self.img_w,
                fy=self.rgb_matrix[1][1],
            )

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        # pipeline.setXLinkChunkSize(0)  # << important to increase throughtput!!! ?
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        self.pd_input_length = 128

        if self.input_type == "rgb":
            # Create ColorCamera node for RGB input
            print("Creating Color Camera")
            # _pgraph_ name 
            cam = pipeline.createColorCamera()
            if self.resolution[0] == 1920:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            if self.focus is not None:
                cam.initialControl.setManualFocus(self.focus)

            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)

        # Create face manager script node
        face_manager_script = pipeline.create(dai.node.Script)
        face_manager_script.setScript(self.build_face_manager_script())
        face_manager_script.setProcessor(dai.ProcessorType.LEON_CSS)
        # Connect ColorCamera or XLinkIn to face manager script based on input type
        if self.input_type == "rgb":
            cam.preview.link(face_manager_script.inputs["cam_in"])
            face_manager_script.inputs["cam_in"].setQueueSize(1)
            face_manager_script.inputs["cam_in"].setBlocking(False)
        else:
            host_to_face_manager_in = pipeline.createXLinkIn()
            host_to_face_manager_in.setStreamName("face_manager_in")
            host_to_face_manager_in.out.link(face_manager_script.inputs["cam_in"])
            

        # Connect ColorCamera output to XLinkOut for RGB input
        if self.input_type == "rgb":
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            # cam_out.input.setQueueSize(1)
            # cam_out.input.setBlocking(False)
            face_manager_script.outputs["cam_out"].link(cam_out.input)

        if self.nb_hands > 0:
            # Define hand manager script node
            hand_manager_script = pipeline.create(dai.node.Script)
            hand_manager_script.setScript(self.build_hand_manager_script())
            hand_manager_script.setProcessor(dai.ProcessorType.LEON_CSS)
            face_manager_script.outputs["hand_manager"].link(hand_manager_script.inputs["face_manager"])

            # Define palm detection pre processing: resize preview to (self.pd_input_length, self.pd_input_length)
            print("Creating Palm Detection pre processing image manip")
            pre_pd_manip = pipeline.create(dai.node.ImageManip)
            pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
            pre_pd_manip.initialConfig.setResizeThumbnail(self.pd_input_length, self.pd_input_length, 0, 0, 0)

            # pre_pd_manip.setWaitForConfigInput(True)
            pre_pd_manip.inputImage.setQueueSize(1)
            hand_manager_script.outputs['pre_pd_manip_frame'].link(pre_pd_manip.inputImage)
            # hand_manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

            # For debugging
            if self.trace & 4:
                pre_pd_manip_out = pipeline.createXLinkOut()
                pre_pd_manip_out.setStreamName("pre_pd_manip_out")
                pre_pd_manip.out.link(pre_pd_manip_out.input)

            # Define palm detection model
            print("Creating Palm Detection Neural Network...")
            pd_nn = pipeline.create(dai.node.NeuralNetwork)
            pd_nn.setBlobPath(self.pd_model)
            pre_pd_manip.out.link(pd_nn.input)
            pd_nn.out.link(hand_manager_script.inputs['from_post_pd_nn'])

            # Define link to send result to host 
            hand_manager_out = pipeline.create(dai.node.XLinkOut)
            hand_manager_out.setStreamName("hand_manager_out")
            hand_manager_script.outputs['host'].link(hand_manager_out.input)
            hand_manager_script.setProcessor(dai.ProcessorType.LEON_CSS)


            # Define hand landmark pre processing image manip
            print("Creating Hand Landmark pre processing image manip") 
            self.hlm_input_length = 224
            pre_hlm_manip = pipeline.create(dai.node.ImageManip)
            pre_hlm_manip.setMaxOutputFrameSize(self.hlm_input_length*self.hlm_input_length*3)
            pre_hlm_manip.setWaitForConfigInput(True)
            pre_hlm_manip.inputImage.setQueueSize(2)

            # For debugging
            if self.trace & 4:
                pre_hlm_manip_out = pipeline.createXLinkOut()
                pre_hlm_manip_out.setStreamName("pre_hlm_manip_out")
                pre_hlm_manip.out.link(pre_hlm_manip_out.input)

            hand_manager_script.outputs['pre_lm_manip_frame'].link(pre_hlm_manip.inputImage)
            hand_manager_script.outputs['pre_lm_manip_cfg'].link(pre_hlm_manip.inputConfig)

            # Define hand landmark model
            print(f"Creating Hand Landmark Neural Network")          
            hlm_nn = pipeline.create(dai.node.NeuralNetwork)
            hlm_nn.setBlobPath(self.hlm_model)
            pre_hlm_manip.out.link(hlm_nn.input)
            hlm_nn.out.link(hand_manager_script.inputs['from_lm_nn'])

        # Define face detection pre-processing image manip
        self.fd_input_length = 128

        print("Creating Face Detection pre processing image manip")
        pre_fd_manip = pipeline.create(dai.node.ImageManip)
        pre_fd_manip.setMaxOutputFrameSize(self.fd_input_length*self.fd_input_length*3)
        # pre_fd_manip.setWaitForConfigInput(True)
        pre_fd_manip.initialConfig.setResizeThumbnail(self.fd_input_length, self.fd_input_length, 0, 0, 0)
        pre_fd_manip.inputImage.setQueueSize(1)
        face_manager_script.outputs['pre_fd_manip_frame'].link(pre_fd_manip.inputImage)
        # face_manager_script.outputs['pre_fd_manip_cfg'].link(pre_fd_manip.inputConfig)

        # For debugging
        if self.trace & 4:
            pre_fd_manip_out = pipeline.createXLinkOut()
            pre_fd_manip_out.setStreamName("pre_fd_manip_out")
            pre_fd_manip.out.link(pre_fd_manip_out.input)

        # Define face detection model
        print("Creating Face Detection Neural Network")
        fd_nn = pipeline.create(dai.node.NeuralNetwork)
        fd_nn.setBlobPath(self.fd_model)
        pre_fd_manip.out.link(fd_nn.input)
        fd_nn.out.link(face_manager_script.inputs['from_post_fd_nn'])

        # Define link to send result to host 
        face_manager_out = pipeline.create(dai.node.XLinkOut)
        face_manager_out.setStreamName("face_manager_out")
        face_manager_script.outputs['host'].link(face_manager_out.input)

        # Define face landmark pre processing image manip
        print("Creating Face Landmark pre processing image manip") 
        self.flm_input_length = 192
        pre_flm_manip = pipeline.create(dai.node.ImageManip)
        pre_flm_manip.setMaxOutputFrameSize(self.flm_input_length*self.flm_input_length*3)
        pre_flm_manip.setWaitForConfigInput(True)
        pre_flm_manip.inputImage.setQueueSize(2)

        # For debugging
        if self.trace & 4:
            pre_flm_manip_out = pipeline.createXLinkOut()
            pre_flm_manip_out.setStreamName("pre_flm_manip_out")
            pre_flm_manip.out.link(pre_flm_manip_out.input)

        face_manager_script.outputs['pre_lm_manip_frame'].link(pre_flm_manip.inputImage)
        face_manager_script.outputs['pre_lm_manip_cfg'].link(pre_flm_manip.inputConfig)

        # Define face landmark model
        print(f"Creating Face Landmark Neural Network")          
        flm_nn = pipeline.create(dai.node.NeuralNetwork)
        flm_nn.setBlobPath(self.flm_model)
        pre_flm_manip.out.link(flm_nn.inputs["lm_input_1"])
        face_manager_script.outputs['sqn_rr'].link(flm_nn.inputs['pp_sqn_rr'])
        face_manager_script.outputs['rot'].link(flm_nn.inputs['pp_rot'])
        flm_nn.out.link(face_manager_script.inputs['from_lm_nn'])

        flm_nn_out = pipeline.create(dai.node.XLinkOut)
        flm_nn_out.setStreamName("flm_nn_out")
        flm_nn.out.link(flm_nn_out.input)

        if self.double_face:
            # Create second set of face landmarks for double-face
            print("Creating Face Landmark pre processing image manip 2") 
            pre_flm_manip2 = pipeline.create(dai.node.ImageManip)
            pre_flm_manip2.setMaxOutputFrameSize(self.flm_input_length*self.flm_input_length*3)
            pre_flm_manip2.setWaitForConfigInput(True)
            pre_flm_manip2.inputImage.setQueueSize(2)

            face_manager_script.outputs['pre_lm_manip_frame2'].link(pre_flm_manip2.inputImage)
            face_manager_script.outputs['pre_lm_manip_cfg2'].link(pre_flm_manip2.inputConfig)
            
            print(f"Creating Face Landmark Neural Network 2")          
            flm_nn2 = pipeline.create(dai.node.NeuralNetwork)
            flm_nn2.setBlobPath(self.flm_model)
            pre_flm_manip2.out.link(flm_nn2.inputs["lm_input_1"])
            face_manager_script.outputs['sqn_rr2'].link(flm_nn2.inputs['pp_sqn_rr'])
            face_manager_script.outputs['rot2'].link(flm_nn2.inputs['pp_rot'])
            flm_nn2.out.link(face_manager_script.inputs['from_lm_nn2'])

            flm_nn2.out.link(flm_nn_out.input)


        if self.xyz:
            # Create MonoCameras, Stereo, and SpatialLocationCalculator nodes for depth calculation
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes")
            # For now, RGB needs fixed focus to properly align with depth.
            # The value used during calibration should be used here
            calib_data = self.device.readCalibration()
            calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
            print(f"RGB calibration lens position: {calib_lens_pos}")
            cam.initialControl.setManualFocus(calib_lens_pos)

            mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
            left = pipeline.createMonoCamera()
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            left.setResolution(mono_resolution)
            left.setFps(self.internal_fps)

            right = pipeline.createMonoCamera()
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            right.setResolution(mono_resolution)
            right.setFps(self.internal_fps)

            stereo = pipeline.createStereoDepth()
            stereo.setConfidenceThreshold(150)
            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setSubpixel(False)  # subpixel True brings latency
            # MEDIAN_OFF necessary in depthai 2.7.2. 
            # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
            # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

            left.out.link(stereo.left)
            right.out.link(stereo.right)    

            depth_out = pipeline.create(dai.node.XLinkOut)
            depth_out.setStreamName("depth_out")
            stereo.depth.link(depth_out.input)

        print("Pipeline created.")
        return pipeline        
    
    def build_hand_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        # Read the template
        with open(HAND_TEMPLATE_MANAGER_SCRIPT_SOLO if self.nb_hands == 1 else HAND_TEMPLATE_MANAGER_SCRIPT_DUO, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#",
                    _TRACE2 = "node.warn" if self.trace & 2 else "#",
                    _lm_score_thresh = self.hlm_score_thresh,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _IF_USE_HANDEDNESS_AVERAGE = "",
                    _single_hand_tolerance_thresh= self.single_hand_tolerance_thresh,
                    _IF_USE_WORLD_LANDMARKS = "" if self.use_world_landmarks else '"""',
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace & 8:
            with open("hand_tmp_code.py", "w") as file:
                file.write(code)

        return code

    def build_face_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        # Read the template
        with open(FACE_TEMPLATE_MANAGER_SCRIPT, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#", # Display a warning if trace is enabled
                    _TRACE2 = "node.warn" if self.trace & 2 else "#", # Display a warning if trace is enabled
                    _with_attention = self.with_attention, 
                    _lm_score_thresh = self.flm_score_thresh, # Score threshold for hand landmarks
                    _pad_h = self.pad_h, # Padding height
                    _img_h = self.img_h, # Image height
                    _img_w = self.img_w, # Image width
                    _frame_size = self.frame_size, # Frame size
                    _crop_w = self.crop_w, # Cropping width
                    _IF_SEND_RGB_TO_HOST = "" if self.input_type == "rgb" else '"""', # Send RGB to host, Use handedness average
                    _track_hands = self.nb_hands > 0, # Tolerance threshold for single hand detection
                    _double_face = self.double_face # Use world coordinates for landmarks
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace & 8:
            with open("face_tmp_code.py", "w") as file:
                file.write(code)

        return code

    def extract_hand_data(self, res, hand_idx):
        # Create an instance of the HandRegion class from the mpu module
        hand = mpu.HandRegion()
        # Extracting rectangular region information
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx] 
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation)
        # Extracting landmark information
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1,3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1,2).astype(np.int)

        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        # Adjust landmark coordinates and rect_points if padding was added to make the image square
        if self.pad_h > 0:
            hand.landmarks[:,1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:,0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # Extract world landmarks if enabled
        if self.use_world_landmarks:
            hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)

        # Recognize gestures if gesture recognition is enabled
        if self.use_gesture: mpu.recognize_gesture(hand)

        return hand

    def extract_face_data(self, res_lm_script, res_lm_nn):
        """
    Extracts face data from the inference results for landmarks (face script and face neural network).

    Parameters:
        - res_lm_script (dict): Inference results containing face-related information from the script.
        - res_lm_nn (NeuralNetworkResult): Inference results containing face-related information from the neural network.

    Returns:
        - face (mpu.Face): Object containing extracted face data.
    """
        # Check if face landmark score is below the threshold, if so, return None
        if self.with_attention:
            lm_score = res_lm_nn.getLayerFp16("lm_conv_faceflag")[0] 
        else:
            lm_score = res_lm_nn.getLayerFp16("lm_score")[0]
        if lm_score < self.flm_score_thresh: return None
        # Create an instance of the Face class from the mpu module
        face = mpu.Face()
        # Extracting rectangular region information
        face.lm_score = lm_score
        face.rect_x_center_a = res_lm_script["rect_center_x"] * self.frame_size
        face.rect_y_center_a = res_lm_script["rect_center_y"] * self.frame_size
        face.rect_w_a = face.rect_h_a = res_lm_script["rect_size"] * self.frame_size
        face.rotation = res_lm_script["rotation"]
        face.rect_points = mpu.rotated_rect_to_points(face.rect_x_center_a, face.rect_y_center_a, face.rect_w_a, face.rect_h_a, face.rotation)
        # Extracting landmark information from the neural network results
        sqn_xy = res_lm_nn.getLayerFp16("pp_sqn_xy")
        sqn_z = res_lm_nn.getLayerFp16("pp_sqn_z")
        rrn_xy = res_lm_nn.getLayerFp16("pp_rrn_xy")
        rrn_z = res_lm_nn.getLayerFp16("pp_rrn_z")

        # If with_attention is enabled, additional processing for different facial zones and iris landmarks
        if self.with_attention:
            # rrn_xy and sqn_xy are the concatenation of 2d landmarks:
            # 468 basic landmarks
            # 80 lips landmarks
            # 71 left eye landmarks
            # 71 right eye landmarks
            # 5 left iris landmarks
            # 5 right iris landmarks
            #
            # rrn_z and sqn_z corresponds to 468 basic landmarks
            
            # face.landmarks = 3D landmarks in the original image in pixels
            # Extract 3D landmarks in the original image in pixels
            lm_xy = (np.array(sqn_xy).reshape(-1,2) * self.frame_size).astype(np.int)
            lm_zone = {}
            lm_zone["lips"] = lm_xy[468:548]
            lm_zone["left eye"] = lm_xy[548:619]
            lm_zone["right eye"] = lm_xy[619:690]
            lm_zone["left iris"] = lm_xy[690:695]
            lm_zone["right iris"] = lm_xy[695:700]
            for zone in ["lips", "left eye", "right eye"]:
                idx_map = mpu.XY_REFINEMENT_IDX_MAP[zone]
                np.put_along_axis(lm_xy, idx_map, lm_zone[zone], axis=0)
            lm_xy[468:473] = lm_zone["left iris"]
            lm_xy[473:478] = lm_zone["right iris"]
            lm_xy = lm_xy[:478]
            lm_z = (np.array(sqn_z) * self.frame_size)
            left_iris_z = np.mean(lm_z[mpu.Z_REFINEMENT_IDX_MAP['left iris']])
            right_iris_z = np.mean(lm_z[mpu.Z_REFINEMENT_IDX_MAP['right iris']])
            lm_z = np.hstack((lm_z, np.repeat([left_iris_z], 5), np.repeat([right_iris_z], 5))).reshape(-1, 1)
            face.landmarks = np.hstack((lm_xy, lm_z)).astype(np.int)

            # face.norm_landmarks = 3D landmarks inside the rotated rectangle, values in [0..1]
            # Extract 3D normalized landmarks inside the rotated rectangle, values in [0..1]
            nlm_xy = np.array(rrn_xy).reshape(-1,2)
            nlm_zone = {}
            nlm_zone["lips"] = nlm_xy[468:548]
            nlm_zone["left eye"] = nlm_xy[548:619]
            nlm_zone["right eye"] = nlm_xy[619:690]
            nlm_zone["left iris"] = nlm_xy[690:695]
            nlm_zone["right iris"] = nlm_xy[695:700]
            for zone in ["lips", "left eye", "right eye"]:
                idx_map = mpu.XY_REFINEMENT_IDX_MAP[zone]
                np.put_along_axis(nlm_xy, idx_map, nlm_zone[zone], axis=0)
            nlm_xy[468:473] = nlm_zone["left iris"]
            nlm_xy[473:478] = nlm_zone["right iris"]
            nlm_xy = nlm_xy[:478]
            nlm_z = np.array(rrn_z)
            left_iris_z = np.mean(nlm_z[mpu.Z_REFINEMENT_IDX_MAP['left iris']])
            right_iris_z = np.mean(nlm_z[mpu.Z_REFINEMENT_IDX_MAP['right iris']])
            nlm_z = np.hstack((nlm_z, np.repeat([left_iris_z], 5), np.repeat([right_iris_z], 5))).reshape(-1, 1)
            face.norm_landmarks = np.hstack((nlm_xy, nlm_z))

        else:
            # If with_attention is not enabled, extract 3D normalized landmarks and landmarks in pixels
            face.norm_landmarks = np.hstack((np.array(rrn_xy).reshape(-1,2), np.array(rrn_z).reshape(-1,1)))
            lm_xy = (np.array(sqn_xy) * self.frame_size).reshape(-1,2)
            lm_z = (np.array(sqn_z) * self.frame_size).reshape(-1, 1)
            face.landmarks = np.hstack((lm_xy, lm_z)).astype(np.int)

        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        # Adjust landmark coordinates and rect_points if padding was applied to make the image square
        if self.pad_h > 0:
            face.landmarks[:,1] -= self.pad_h
            for i in range(len(face.rect_points)):
                face.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            face.landmarks[:,0] -= self.pad_w
            for i in range(len(face.rect_points)):
                face.rect_points[i][0] -= self.pad_w

        # If face pose estimation is enabled, calculate metric landmarks and pose vectors
        if self.use_face_pose:
            screen_landmarks = (face.landmarks / np.array([self.img_w, self.img_h, self.img_w])).T
            face.metric_landmarks, face.pose_transform_mat = get_metric_landmarks(screen_landmarks, self.pcf)
            # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
            face.pose_transform_mat[1:3, :] = -face.pose_transform_mat[1:3, :]
            face.pose_rotation_vector, _ = cv2.Rodrigues(face.pose_transform_mat[:3, :3])
            face.pose_translation_vector = face.pose_transform_mat[:3, 3, None]
        return face

    def next_frame(self):
        """
    Processes the next frame from the video source.

    Returns:
        - video_frame (np.ndarray): Processed video frame.
        - faces (list): List of Face objects representing detected faces.
        - hands (list): List of HandRegion objects representing detected hands.
    """
        # Handling special case in double face mode with non-RGB input
        if self.double_face and self.input_type != "rgb" and self.seq_num == 0:
            # If double face mode, non-RGB input, and first iteration, send two frames to ensure parallel inferences
            # Because there are 2 inferences running in parallel in double face mode, we need to send 2 frames on the first loop iteration
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
            self.prev_video_frame = video_frame
           
            # Create ImgFrame and send to face_manager
            frame = dai.ImgFrame()
            frame.setType(dai.ImgFrame.Type.BGR888p)
            h,w = video_frame.shape[:2]
            frame.setWidth(w)
            frame.setHeight(h)
            frame.setData(video_frame.transpose(2,0,1).flatten())
            self.q_face_manager_in.send(frame)

        # Increment sequence number and update FPS
        self.seq_num += 1
        self.fps.update()
        # Handling RGB input or sending current frame to face_manager
        if self.input_type == "rgb":
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()  
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
           
            # Create ImgFrame and send to face_manager
            frame = dai.ImgFrame()
            frame.setType(dai.ImgFrame.Type.BGR888p)
            h,w = video_frame.shape[:2]
            frame.setWidth(w)
            frame.setHeight(h)
            frame.setData(video_frame.transpose(2,0,1).flatten())
            self.q_face_manager_in.send(frame)

        # For debugging : # Create ImgFrame and send to face_manager
        if self.trace & 4:
            # Create ImgFrame and send to face_manager
            if self.nb_hands > 0:
                pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
                if pre_pd_manip:
                    pre_pd_manip = pre_pd_manip.getCvFrame()
                    cv2.imshow("pre_pd_manip", pre_pd_manip)
                pre_hlm_manip = self.q_pre_hlm_manip_out.tryGet()
                if pre_hlm_manip:
                    pre_hlm_manip = pre_hlm_manip.getCvFrame()
                    cv2.imshow("pre_hlm_manip", pre_hlm_manip)
            pre_fd_manip = self.q_pre_fd_manip_out.tryGet()
            if pre_fd_manip:
                pre_fd_manip = pre_fd_manip.getCvFrame()
                cv2.imshow("pre_fd_manip", pre_fd_manip)
            pre_flm_manip = self.q_pre_flm_manip_out.tryGet()
            if pre_flm_manip:
                pre_flm_manip = pre_flm_manip.getCvFrame()
                cv2.imshow("pre_flm_manip", pre_flm_manip)

        # Get result from device
        hands = []
        if self.nb_hands > 0:
            res = marshal.loads(self.q_hand_manager_out.get().getData())
            for i in range(len(res.get("lm_score",[]))):
                hand = self.extract_hand_data(res, i)
                hands.append(hand)

        # Get face information from face_manager output
        res_lm_script = marshal.loads(self.q_face_manager_out.get().getData())
        status = res_lm_script["status"]
        faces = []
        # Handle different statuses
        # status = 0 means the face detector has run but detected no face
        # status = 1 means face_manager_script has initiated an face landmark inference,
        #            and the face landmark NN will send directly the result here, on the host
        if status == 1:
            res_lm_nn = self.q_flm_nn_out.get()
            face = self.extract_face_data(res_lm_script, res_lm_nn)
            if face is not None: faces.append(face)

        # Handle depth information if xyz is enabled
        if self.xyz:
            t = now()
            in_depth_msgs = self.q_depth_out.getAll()
            self.depth_sync.add(in_depth_msgs)
            synced_depth_msg = self.depth_sync.get(in_video) 
            frame_depth = synced_depth_msg.getFrame()

            # Extract 3D coordinates for hands and faces
        if self.nb_hands > 0:
            for hand in hands:
                hand.xyz, hand.xyz_zone = self.spatial_calc.get_xyz(frame_depth, hand.landmarks[0])
                # Check distance between mouth and hand landmarks
                distance_threshold = 50  # Set your desired threshold
                distance_mouth_hand = dist.euclidean(face.landmarks[9, :2], hand.landmarks[0, :2])
                if distance_mouth_hand > distance_threshold:
                    print("Distance Threshold Exceeded! Displaying Message...")
                    # You can display a message or take other actions here
                    
        for face in faces:
            face.xyz, face.xyz_zone = self.spatial_calc.get_xyz(frame_depth, face.landmarks[9, :2])

            # !!! The 4 lines below are for disparity (not depth)
            # frame_depth = (frame_depth * 255. / self.max_disparity).astype(np.uint8)
            # frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_HOT)
            # frame_depth = np.ascontiguousarray(frame_depth)
            # cv2.imshow("depth", frame_depth)
            # Extract 3D coordinates for hands and faces
            if self.nb_hands > 0:
                for hand in hands:
                    hand.xyz, hand.xyz_zone = self.spatial_calc.get_xyz(frame_depth, hand.landmarks[0])
            for face in faces:
                face.xyz, face.xyz_zone = self.spatial_calc.get_xyz(frame_depth, face.landmarks[9,:2])

        # Handle double face mode with non-RGB input
        if self.double_face and self.input_type != "rgb":
            video_frame, self.prev_video_frame = self.prev_video_frame, video_frame
        return video_frame, faces, hands


    def exit(self):
        """
        Closes the device and prints the average frames per second.
        """
        self.device.close()
        print(f"FPS : {self.fps.get_global():.1f} f/s")
            