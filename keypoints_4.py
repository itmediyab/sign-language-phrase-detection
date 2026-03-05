import os
import cv2
import numpy as np
import mediapipe as mp

# ================= CONFIG =================
base_dir = os.getcwd()
VIDEO_PATH = os.path.join(base_dir, "videos")
DATA_PATH = os.path.join(base_dir, "keypoints_data2")
no_sequences = 30   # frames per video

# Automatically get action names from video folders
actions = np.array(os.listdir(VIDEO_PATH))

# =============== MEDIAPIPE SETUP ===============
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.array(
        [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array(
        [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([lh, rh])

# =============== MAIN LOOP ===============
cv2.namedWindow('Extracting Keypoints', cv2.WINDOW_NORMAL)

for action in actions:
    action_video_path = os.path.join(VIDEO_PATH, action)
    action_data_path = os.path.join(DATA_PATH, action)

    os.makedirs(action_data_path, exist_ok=True)

    # Create 30 sequence folders
    for i in range(no_sequences):
        os.makedirs(os.path.join(action_data_path, str(i)), exist_ok=True)

    video_counter = 0

    for video_file in os.listdir(action_video_path):
        print(f"\n▶ Processing {video_file}")
        full_video_path = os.path.join(action_video_path, video_file)

        cap = cv2.VideoCapture(full_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < no_sequences:
            print("⚠ Skipped - not enough frames")
            continue

        frame_indices = np.linspace(0, total_frames - 1, no_sequences, dtype=int)

        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:

            for i, frame_index in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()

                if not ret:
                    print(f"⚠ Could not read frame {frame_index}")
                    continue

                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks on the frame
                draw_landmarks(image, results)
                
                # Add text overlay showing progress
                cv2.rectangle(image, (0, 0), (640, 60), (0, 0, 0), -1)
                cv2.putText(image, f"Action: {action}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Frame: {i+1}/{no_sequences} | Video: {video_counter+1}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Resize frame to fit screen while maintaining aspect ratio
                screen_w, screen_h = 1280, 720
                h, w = image.shape[:2]
                scale = min(screen_w / w, screen_h / h)
                display_w, display_h = int(w * scale), int(h * scale)
                display_frame = cv2.resize(image, (display_w, display_h))

                # Resize window to match the scaled frame exactly
                cv2.resizeWindow('Extracting Keypoints', display_w, display_h)
                cv2.imshow('Extracting Keypoints', display_frame)
                cv2.waitKey(50)  # Display for 50ms (adjust for speed)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(action_data_path, str(i), f"{video_counter}.npy")
                np.save(npy_path, keypoints)

        cap.release()
        video_counter += 1

cv2.destroyAllWindows()
print("\n✅ All videos processed successfully!")