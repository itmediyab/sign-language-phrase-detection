import cv2
import mediapipe as mp
import numpy as np
import os

# ================= CONFIG =================
INPUT_FOLDER = "videos-2"
OUTPUT_FOLDER = "skeletal-vids-final"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= MEDIAPIPE =================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def convert_to_skeleton(input_path, output_path):
    """Convert a single video to skeleton format"""
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open {input_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # ================= DRAWING SPECS =================
    face_color = (192, 192, 192)  # Light Gray
    
    pose_landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
    pose_connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    
    left_hand_landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    left_hand_connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
    
    right_hand_landmark_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
    right_hand_connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
    
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            
            # Black background
            skeleton = np.zeros((height, width, 3), dtype=np.uint8)
            
            # ================= DRAW FACE WITH PROPER FEATURES =================
            if results.face_landmarks:
                h, w = height, width
                face_lm = results.face_landmarks.landmark
                
                # Face oval outline
                face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                
                # Draw face outline
                for i in range(len(face_oval)):
                    start_idx = face_oval[i]
                    end_idx = face_oval[(i + 1) % len(face_oval)]
                    start = face_lm[start_idx]
                    end = face_lm[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(skeleton, start_point, end_point, face_color, 2)
                
                # Left Eye
                left_eye_points = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
                left_eye_coords = np.array([(int(face_lm[i].x * w), int(face_lm[i].y * h)) for i in left_eye_points], np.int32)
                cv2.polylines(skeleton, [left_eye_coords], True, face_color, 2)
                
                # Right Eye
                right_eye_points = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
                right_eye_coords = np.array([(int(face_lm[i].x * w), int(face_lm[i].y * h)) for i in right_eye_points], np.int32)
                cv2.polylines(skeleton, [right_eye_coords], True, face_color, 2)
                
                # Lips (Outer)
                outer_lip_points = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 
                                   375, 321, 405, 314, 17, 84, 181, 91, 146]
                outer_lip_coords = np.array([(int(face_lm[i].x * w), int(face_lm[i].y * h)) for i in outer_lip_points], np.int32)
                cv2.polylines(skeleton, [outer_lip_coords], True, face_color, 2)
                
                # Lips (Inner)
                inner_lip_points = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 
                                   324, 318, 402, 317, 14, 87, 178, 88, 95]
                inner_lip_coords = np.array([(int(face_lm[i].x * w), int(face_lm[i].y * h)) for i in inner_lip_points], np.int32)
                cv2.polylines(skeleton, [inner_lip_coords], True, face_color, 2)
            
            # ================= DRAW POSE WITHOUT WRISTS (RED) =================
            if results.pose_landmarks:
                h, w = height, width
                pose_lm = results.pose_landmarks.landmark
                
                # Define pose connections WITHOUT wrist connections
                # Original pose has connections to wrists (15, 16), we'll exclude those
                pose_connections_no_wrists = [
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
                    (11, 23), (12, 24), (23, 24),  # Torso
                    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
                    (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
                ]
                
                # Draw custom pose connections
                for connection in pose_connections_no_wrists:
                    start_idx, end_idx = connection
                    start = pose_lm[start_idx]
                    end = pose_lm[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(skeleton, start_point, end_point, (0, 0, 255), 2)
                
                # Draw pose landmarks (excluding wrists 15, 16)
                pose_landmarks_to_draw = [11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
                for idx in pose_landmarks_to_draw:
                    landmark = pose_lm[idx]
                    point = (int(landmark.x * w), int(landmark.y * h))
                    cv2.circle(skeleton, point, 3, (0, 0, 255), -1)
            
            # ================= DRAW LEFT HAND (GREEN) =================
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    skeleton,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=left_hand_landmark_spec,
                    connection_drawing_spec=left_hand_connection_spec
                )
            
            # ================= DRAW RIGHT HAND (YELLOW) =================
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    skeleton,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=right_hand_landmark_spec,
                    connection_drawing_spec=right_hand_connection_spec
                )
            
            out.write(skeleton)
            
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"    Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"  ✅ Saved: {output_path}\n")
    return True

# ================= PROCESS ALL VIDEOS =================
print("=" * 70)
print("BATCH CONVERTING VIDEOS TO SKELETON (NO WRIST CLASH)")
print("=" * 70)
print(f"Input folder:  {INPUT_FOLDER}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("=" * 70)
print("\nColor Scheme:")
print("  ⚪ Face: Light Gray (outline + eyes + lips)")
print("  🔴 Body/Pose: Red (no wrist overlap)")
print("  🟢 Left Hand: Green")
print("  🟡 Right Hand: Yellow")
print("=" * 70)

if not os.path.exists(INPUT_FOLDER):
    print(f"\n❌ ERROR: Folder not found: {INPUT_FOLDER}")
    exit()

# Get all video files
video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]

if not video_files:
    print(f"\n❌ ERROR: No video files found in {INPUT_FOLDER}")
    exit()

print(f"\nFound {len(video_files)} video(s) to process\n")

successful = 0
failed = 0

for i, video_file in enumerate(video_files, 1):
    print(f"[{i}/{len(video_files)}] Processing: {video_file}")
    
    input_path = os.path.join(INPUT_FOLDER, video_file)
    
    output_filename = os.path.splitext(video_file)[0] + "_skeleton.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    
    if convert_to_skeleton(input_path, output_path):
        successful += 1
    else:
        failed += 1

print("=" * 70)
print("CONVERSION COMPLETE!")
print("=" * 70)
print(f"✅ Successful: {successful}")
print(f"❌ Failed: {failed}")
print(f"📁 Output folder: {OUTPUT_FOLDER}")
print("=" * 70)