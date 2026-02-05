Sign Language Phrase Detection
- This project detects sign language phrases for a patient–doctor hospital appointment scenario using machine learning.
- It aims to assist communication in hospital settings by detecting commonly used appointment phrases in real time.
- It extracts hand keypoints in csv format using Mediapipe Holistic, trains a model and performs live webcam detection.

Dataset
- 21 hospital related phrases were collected, with 20-30 videos per phrase.
- Videos were recorded for each phrase, which were broken down into frames.
- Hand keypoints were extracted from the video frames.
- The extracted keypoints are stored as CSV datasets.
- These CSV files were used to train the machine learning model.

