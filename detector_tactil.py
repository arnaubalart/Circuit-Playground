import freenect
import cv2
import numpy as np

def get_depth():
    """Fetches a depth frame from the Kinect."""
    array, _ = freenect.sync_get_depth()
    return array

def get_min_depth_in_roi(frame, cx, cy, size=20):
    """
    Looks at a small square (Region of Interest - ROI) in the center.
    Returns the depth of the object CLOSEST to the camera (the fingertip).
    """
    half = size // 2
    roi = frame[cy - half : cy + half, cx - half : cx + half]
    
    # Filter out Kinect error pixels (2047 means no data)
    valid_pixels = roi[roi < 2047]
    
    if len(valid_pixels) == 0:
        return 2047
    
    return int(np.min(valid_pixels))

def main():
    print("Starting Kinect Interactive Touch System...")
    print("Please look at the popup window for instructions.")
    
    # Center coordinates of the Kinect vision
    cx, cy = 320, 240 
    
    # System States:
    # 0 = Calibrating Table
    # 1 = Calibrating Finger
    # 2 = Running Mode
    state = 0 
    
    table_depth = 0
    finger_depth = 0
    touch_threshold = 0
    
    # Tolerance margin: +/- units from the exact finger depth to still count as a "touch"
    tolerance = 4 

    while True:
        frame = get_depth()
        if frame is None:
            continue

        # Convert raw depth data to a visible grayscale image
        vis = frame.astype(np.uint8)
        # Convert to BGR color space so we can draw colored text and boxes on it
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # Get the distance of whatever is inside the center box
        current_depth = get_min_depth_in_roi(frame, cx, cy)

        # ---------------------------------------------------------
        # STATE 0: TABLE CALIBRATION
        # ---------------------------------------------------------
        if state == 0:
            cv2.putText(vis, "STEP 1: Clear the table completely.", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(vis, "Press 'c' to capture the table.", (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw a Red box
            cv2.rectangle(vis, (cx-10, cy-10), (cx+10, cy+10), (0, 0, 255), 2)

        # ---------------------------------------------------------
        # STATE 1: FINGER CALIBRATION
        # ---------------------------------------------------------
        elif state == 1:
            cv2.putText(vis, "STEP 2: Place finger inside the box", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(vis, "TOUCH the paper and press 'c'.", (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            # Draw a Yellow box
            cv2.rectangle(vis, (cx-10, cy-10), (cx+10, cy+10), (0, 255, 255), 2)

        # ---------------------------------------------------------
        # STATE 2: RUNNING (DETECTION MODE)
        # ---------------------------------------------------------
        elif state == 2:
            diff_from_table = table_depth - current_depth
            status_text = "EMPTY TABLE"
            status_color = (100, 100, 100) # Gray

            if current_depth < 2047:
                # If the difference is much bigger than the threshold, finger is high in the air
                if diff_from_table > (touch_threshold + tolerance):
                    status_text = "HOVERING (AIR)"
                    status_color = (255, 0, 0) # Blue
                
                # If the difference matches our calibrated finger thickness (within tolerance)
                elif (touch_threshold - tolerance) <= diff_from_table <= (touch_threshold + tolerance):
                    status_text = "TOUCH DETECTED!"
                    status_color = (0, 255, 0) # Green
                    
            # Draw UI
            cv2.putText(vis, f"STATUS: {status_text}", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(vis, f"Current Diff: {diff_from_table} | Target: {touch_threshold}", (15, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Draw colored box responding to the status
            cv2.rectangle(vis, (cx-10, cy-10), (cx+10, cy+10), status_color, 2)


        cv2.imshow("Kinect Smart Calibration", vis)
        
        # Keyboard Input detection
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            print("Quitting program...")
            break
            
        elif key == ord('c'):
            # Logic for pressing 'c' depends on the current state
            if state == 0:
                if current_depth < 2047:
                    table_depth = current_depth
                    state = 1
                    print(f"[OK] Table depth saved: {table_depth}")
                else:
                    print("[ERROR] Cannot see table. Check Kinect position.")
                    
            elif state == 1:
                if current_depth < 2047:
                    finger_depth = current_depth
                    touch_threshold = table_depth - finger_depth
                    
                    if touch_threshold <= 0:
                        print("[WARNING] Finger seems to be at the same level as the table. Try again.")
                    else:
                        state = 2
                        print(f"[OK] Finger depth saved: {finger_depth}")
                        print(f"[OK] System calibrated! Touch Thickness = {touch_threshold}")

if __name__ == "__main__":
    main()
