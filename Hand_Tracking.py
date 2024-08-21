import cv2
import mediapipe as mp
import time

# Initialize MediaPipe components
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

PTime = 0  # Previous time

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()  # Capture frame-by-frame
    if not success:
        break  # If frame is not captured correctly, exit the loop

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Optionally, you can draw points or circles at landmarks here if needed

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - PTime) if (cTime - PTime) != 0 else 0
    PTime = cTime

    # Display FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("img", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
