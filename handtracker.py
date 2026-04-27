# import necessary libraries
import freenect
import cv2 as cv
import numpy as np
import time
# mediapipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants for depth normalization
MIN_DEPTH = 400  # Minimum depth in millimeters to consider as a valid hand detection
MAX_DEPTH = 8000 # Maximum depth in millimeters 
# Center coordinates of the Kinect vision
cx, cy = 320, 240 

# Helper functions
def normalize_depth_frame(depth_frame):
    # Normalize the depth frame to the range [0, 255] for visualization
    depth_frame = depth_frame.astype(np.float32)
    depth_frame = (depth_frame - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH) * 255
    return depth_frame.astype(np.uint8)

def hand_tracker(vid_frame, detector):
    # Placeholder for hand tracking logic
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=vid_frame)
    result = detector.detect_for_video(mp_image, timestamp_ms=int(time.time() * 1000))

    if result.hand_landmarks:
        landmark = result.hand_landmarks[0][8]
        x = int(landmark.x * vid_frame.shape[1])
        y = int(landmark.y * vid_frame.shape[0])
        coord = (x, y)
        return coord
    
    return None

def webcam_test(detector):
    # Init video capture
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera is not opened")
        exit()

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        coord = hand_tracker(rgb_frame, detector)
        if coord is not None:
            # Draw a circle at the detected fingertip position
            cv.circle(rgb_frame, coord, 10, (0, 255, 0), -1)

        # Dislpay the resulting frame
        cv.imshow('Hand Tracker', rgb_frame)
        
        # Wait for 'q' key to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup after the loop
    cap.release()
    cv.destroyAllWindows()


# Returns a mediapipe hand tracker object
def init_mediapipe_hand_tracker():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1, 
        running_mode=vision.RunningMode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options)

def is_touching(finger_depth, table_depth, tolerance):
    return abs(finger_depth - table_depth) <= tolerance

def hand_depth(detector):
    table_depth = 0
    tolerance = 4

    while True:
        # Fetch frames from Kinect
        depth_frame, _ = freenect.sync_get_depth()  # Initialize depth stream
        vid_frame, _ = freenect.sync_get_video()  # Initialize color stream

        if depth_frame is None or vid_frame is None:
            continue

        # Normalize depth frame for visualization
        depth_vis = normalize_depth_frame(depth_frame)

        # Convert to BGR for OpenCV display
        depth_vis = cv.cvtColor(depth_vis, cv.COLOR_GRAY2BGR)

        # Hand tracker logic:
        coord = hand_tracker(vid_frame, detector)
        if coord is not None:
            x, y = coord
            depth_value = depth_frame[y, x]  # value in millimeters
            is_touching_table = is_touching(depth_value, table_depth, tolerance)
            print(f"Depth at fingertip: {depth_value} mm, Table depth: {table_depth} mm, Touching: {is_touching_table}")

            # Draw a circle at the detected fingertip position
            cv.circle(depth_vis, coord, 10, (0, 255, 0), -1)

        # Dislpay the resulting frame
        cv.imshow('Hand Tracker', depth_vis)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv.waitKey(1) & 0xFF == ord('c'):
            print("Calibrating table depth...")
            table_depth = depth_frame[cy, cx]
            print(f"Table depth set to: {table_depth} mm")
    
    cv.destroyAllWindows()


# Main program loop
def main():
    detector = init_mediapipe_hand_tracker()
    hand_depth(detector)

# Call program
main()