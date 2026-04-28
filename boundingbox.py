import cv2
import numpy as np

def nothing(x):
    pass

def order_points(pts):
    """
    Orders 4 points in the order: top-left, top-right, bottom-right, bottom-left.
    This is essential for the Perspective Transform.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has the smallest sum, bottom-right has the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has the smallest difference, bottom-left has the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def main():
    # SETUP: Update the IP with your Phone's Hotspot IP
    url = "http://192.168.0.208:8080/video" 
    cap = cv2.VideoCapture(url)

    # Control window
    cv2.namedWindow("Settings")
    cv2.createTrackbar("Threshold", "Settings", 150, 255, nothing)
    cv2.createTrackbar("Min Area", "Settings", 5000, 50000, nothing)

    print("Scanner started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. PRE-PROCESSING
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        thresh_val = cv2.getTrackbarPos("Threshold", "Settings")
        min_area = cv2.getTrackbarPos("Min Area", "Settings")

        # 2. SEGMENTATION
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

        # 3. CONTOUR DETECTION
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                if len(approx) == 4:
                    # Draw on the main frame
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
                    
                    # --- NEW: PERSPECTIVE TRANSFORM (WARP) ---
                    # Reorganize points for the transform
                    pts = approx.reshape(4, 2)
                    rect = order_points(pts)
                    (tl, tr, br, bl) = rect

                    # Define the dimensions of the output scanned image (A4 ratio approx)
                    width, height = 400, 600
                    dst = np.array([
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1],
                        [0, height - 1]], dtype="float32")

                    # Calculate the Perspective Transform matrix
                    M = cv2.getPerspectiveTransform(rect, dst)
                    # Apply the matrix to get the "Bird's Eye View"
                    warped = cv2.warpPerspective(frame, M, (width, height))

                    # Show the result in a separate window
                    cv2.imshow("3. Scanned Document", warped)
                    break # Found the paper, stop looking for other contours

        # 4. DISPLAY
        cv2.imshow("1. Original Feed", frame)
        cv2.imshow("2. Computer Vision Debug", thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()