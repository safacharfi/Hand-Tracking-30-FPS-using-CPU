# Hand Detection and Keypoint Recognition using MediaPipe

This project demonstrates how to use MediaPipe to recognize hand keypoints in real-time using a webcam. MediaPipe returns a total of 21 keypoints for each detected hand.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Initial Models for MediaPipe](#initial-models-for-mediapipe)
- [Read Frames from a Webcam](#read-frames-from-a-webcam)
- [Detect Hand Keypoints](#detect-hand-keypoints)
- [Convert Frame to RGB Format](#convert-frame-to-rgb-format)
- [Example Output](#example-output)
- [License](#license)

## Overview

In this project, we'll use MediaPipe to detect hands in real-time and draw landmarks (key points) on the detected hands. MediaPipe provides a total of 21 key points for each detected hand, as shown in the image below:

![image](https://github.com/user-attachments/assets/ad1f7941-db47-40cf-b6cf-9ee4040c72b0)

## Setup

To get started, ensure you have the following dependencies installed:

```bash
pip install opencv-python mediapipe
```

## Initial Models for MediaPipe

We first initialize the MediaPipe hands model and the drawing utilities:

```python
import mediapipe as mp

self.handsMp = mp.solutions.hands
self.hands = self.handsMp.Hands()
self.mpDraw = mp.solutions.drawing_utils
```

## Read Frames from a Webcam

We use OpenCV to read frames from the webcam. The code below captures video from the default webcam and sets the frame width and height:

```python
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Detect Hand Keypoints

We then process each frame to detect the hand keypoints and draw them on the frame. Here are the functions to find fingers and their positions:

```python
def findFingers(self, frame, draw=True):
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)  
    if self.results.multi_hand_landmarks: 
        for handLms in self.results.multi_hand_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
    return frame

def findPosition(self, frame, handNo=0, draw=True):
    xList = []
    yList = []
    bbox = []
    self.lmsList = []
    if self.results.multi_hand_landmarks:
        myHand = self.results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark):
            h, w, c = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            self.lmsList.append([id, cx, cy])
            if draw:
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = xmin, ymin, xmax, ymax
        if draw:
            cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                          (0, 255, 0), 2)

    return self.lmsList, bbox
```

## Convert Frame to RGB Format

Since MediaPipe works with RGB images, but OpenCV reads images in BGR format, we need to convert the frame from BGR to RGB using `cv2.cvtColor()`:

```python
imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

## Example Output

Once everything is set up, the code should detect and draw key points on the hand in real-time, as shown in this picture 


![image](https://github.com/user-attachments/assets/754e1f41-53f7-4b74-a13a-20c7b96b1dd5)
