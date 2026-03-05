import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp

# ================= CONFIG =================
MODEL_PATH = "model.h5"
DATA_PATH = "keypoints_data2"
SEQUENCE_LENGTH = 30
THRESHOLD = 0.7
MIN_FRAMES = 10

# ================= LOAD MODEL =================
model = load_model(MODEL_PATH)

actions = np.array(os.listdir(DATA_PATH))
print("Actions loaded:", actions)

# ================= MEDIAPIPE SETUP =================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array(
        [[res.x, res.y, res.z]
         for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array(
        [[res.x, res.y, res.z]
         for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([lh, rh])

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def hands_detected(results):
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    sequence = np.array(sequence)
    current_length = len(sequence)
    if current_length == target_length:
        return sequence
    indices = np.linspace(0, current_length - 1, target_length, dtype=int)
    return sequence[indices]

# ================= WEBCAM LOOP =================
cap = cv2.VideoCapture(0)

sequence = []
sentence = ""
is_success = False
is_recording = False

window_name = 'Sign Language Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("🎥 Starting webcam in fullscreen... Press 'q' or 'ESC' to quit")

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        hands_visible = hands_detected(results)

        if hands_visible:
            if not is_recording:
                is_recording = True
                sequence = []
                print("👋 Hands detected - recording gesture...")

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

        else:
            if is_recording:
                if len(sequence) >= MIN_FRAMES:
                    resampled = resample_sequence(sequence, SEQUENCE_LENGTH)
                    res = model.predict(np.expand_dims(resampled, axis=0), verbose=0)[0]
                    action_idx = np.argmax(res)
                    confidence = res[action_idx]

                    if confidence > THRESHOLD:
                        sentence = actions[action_idx]
                        is_success = True
                        print(f"✅ Gesture: {sentence} | {len(sequence)} frames → resampled to {SEQUENCE_LENGTH} | confidence: {confidence:.2f}")
                    else:
                        sentence = "Could not detect gesture"
                        is_success = False
                        print(f"⚠️ Low confidence ({confidence:.2f})")
                else:
                    sentence = "Gesture too short"
                    is_success = False
                    print(f"⚠️ Too few frames ({len(sequence)}) — minimum is {MIN_FRAMES}")

                is_recording = False
                sequence = []

        # ===== DISPLAY UI =====
        h, w = image.shape[:2]

        bar_height = int(h * 0.15)
        cv2.rectangle(image, (0, 0), (w, bar_height), (0, 0, 0), -1)

        font_scale = w / 800.0
        thickness = max(2, int(font_scale * 2))

        if is_recording:
            status_text = f"Recording... ({len(sequence)} frames collected)"
            status_color = (0, 165, 255)  # Orange
        else:
            status_text = "Show hands to start"
            status_color = (255, 255, 255)

        cv2.putText(image, status_text, (10, int(bar_height * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, status_color, thickness)

        if not is_recording and sentence:
            cv2.putText(image, "Detected:", (10, int(bar_height * 0.6)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (255, 255, 255), thickness)

            # Green for success, red for failure
            sentence_color = (0, 255, 0) if is_success else (0, 0, 255)
            cv2.putText(image, sentence, (10, int(bar_height * 0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, sentence_color, thickness + 1)

        cv2.putText(image, "Press 'q' or 'ESC' to quit", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), thickness - 1)

        cv2.imshow(window_name, image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break

cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped")