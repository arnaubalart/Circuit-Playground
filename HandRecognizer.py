import cv2
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

# Inizializzazione modello
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# configuration for full screen
window_name = "Scrivania Virtuale"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1) # mirror effect for selfie view
    
    # convertion to RGB for mediapipe processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    #Text overlay for user instructions
    # Parameter: (image, text, position, font, fontScale, color, thickness, lineType)
    cv2.putText(image, "Press Q or ESC to exit", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(window_name, image)

    # Check for exit key
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27: # 27 is the ESC key
        break

cap.release()
cv2.destroyAllWindows()