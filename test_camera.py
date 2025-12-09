import cv2
import mediapipe as mp

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# This initializes the hand detector
# static_image_mode=False: Optimizes for video (faster)
# max_num_hands=1: We only want to detect one hand for now
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# 2. Open Webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("✅ Camera started! Press 'q' to exit.")

while True:
    # Read a frame from the camera
    success, img = cap.read()
    if not success:
        break

    # 3. Flip the image (mirror effect) and convert to RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. Process the image to find hands
    results = hands.process(img_rgb)

    # 5. If a hand is found, draw the skeleton
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the 21 points and the lines connecting them
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the window
    cv2.imshow("Stage 1: Hand Tracking Test", img)

    # Press 'q' to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
