# Computer Vision: Hand-Tracked Virtual Diamond

## 📌 Overview
This repository features an interactive computer vision application that allows users to manipulate a virtual diamond on-screen using real-time hand gestures. By leveraging a webcam feed, the program maps physical hand movements to on-screen coordinates, creating a seamless, touchless interface.

## ⚙️ Core Technology
* **Hand Tracking:** Utilizes Google's **MediaPipe** machine learning pipeline (`hand_landmarker.task`) to detect and continuously map 3D hand landmarks in real-time.
* **Image Processing:** Implements **OpenCV** to capture the live video feed, render the virtual diamond, and process the frame data.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries/Frameworks:** OpenCV (`cv2`), MediaPipe, NumPy
* **Input:** Real-time webcam video stream

## 📂 Project Demonstration
To see the computer vision tracking in action, please view the attached demo video:
* `WATCH ME -Hand Diamond Demo.mp4`

## 🚀 How to Run
1. Clone this repository to your local machine.
2. Install the necessary computer vision dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
