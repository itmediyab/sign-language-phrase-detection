import cv2
import numpy as np
import os
import speech_recognition as sr
from fuzzywuzzy import fuzz

# ================= CONFIG =================
VIDEO_FOLDER = "skeletal-vids-final"  # Folder with skeleton videos
SIMILARITY_THRESHOLD = 70  # Fuzzy matching threshold (0-100)

# ================= LOAD AVAILABLE SIGNS =================
available_signs = {}

# Load all skeleton videos directly from folder
if os.path.exists(VIDEO_FOLDER):
    for video_file in os.listdir(VIDEO_FOLDER):
        if video_file.endswith(('.mp4', '.avi', '.mov')):
            # Remove "_skeleton" suffix and file extension to get action name
            action_name = video_file.replace('_skeleton.mp4', '').replace('_skeleton.avi', '').replace('_skeleton.mov', '')
            video_path = os.path.join(VIDEO_FOLDER, video_file)
            available_signs[action_name] = video_path

print("=" * 70)
print("📚 AVAILABLE SKELETON SIGNS")
print("=" * 70)
for sign in sorted(available_signs.keys()):
    print(f"  - {sign}")
print("=" * 70)

if not available_signs:
    print("\n❌ ERROR: No skeleton videos found in folder!")
    print(f"   Check that '{VIDEO_FOLDER}' contains videos")
    exit()

# ================= SPEECH RECOGNITION =================
recognizer = sr.Recognizer()
microphone = sr.Microphone()

def listen_for_speech():
    """Listen to microphone and return recognized text"""
    with microphone as source:
        print("\n🎤 Listening... (speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("🔄 Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"✅ Recognized: '{text}'")
            return text.lower()
        except sr.WaitTimeoutError:
            print("⏱️ No speech detected")
            return None
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None

# ================= TEXT TO SIGN MATCHING =================
def find_best_match(text, available_signs):
    """Find the best matching sign for the given text using fuzzy matching"""
    best_match = None
    best_score = 0
    
    # Clean up the text (remove underscores, lowercase)
    text_clean = text.lower().replace("_", " ").strip()
    
    for sign_name in available_signs.keys():
        sign_clean = sign_name.lower().replace("_", " ").strip()
        
        # Calculate similarity score
        score = fuzz.ratio(text_clean, sign_clean)
        
        # Boost score if text is contained in sign name
        if text_clean in sign_clean or sign_clean in text_clean:
            score = max(score, 85)
        
        # Boost score for partial matches
        partial_score = fuzz.partial_ratio(text_clean, sign_clean)
        score = max(score, partial_score)
        
        if score > best_score:
            best_score = score
            best_match = sign_name
    
    if best_score >= SIMILARITY_THRESHOLD:
        return best_match, best_score
    else:
        return None, best_score

# ================= VIDEO PLAYBACK =================
def play_sign_video(video_path, sign_name):
    """Play the skeleton sign video in fullscreen"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_delay = int(1000 / fps) if fps > 0 else 33
    
    # Create fullscreen window
    window_name = 'Skeleton Sign Language'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(f"▶️ Playing sign: {sign_name}")
    
    # Play video in loop 2 times for clarity
    for loop in range(2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add text overlay
            h, w = frame.shape[:2]
            font_scale = w / 800.0
            thickness = max(2, int(font_scale * 2))
            
            # Black bar at top
            cv2.rectangle(frame, (0, 0), (w, int(h * 0.12)), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {sign_name.replace('_', ' ')}", 
                       (10, int(h * 0.07)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.9, (0, 255, 0), thickness)
            
            # Loop indicator
            cv2.putText(frame, f"Loop {loop + 1}/2", 
                       (w - int(w * 0.2), int(h * 0.07)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.6, (255, 255, 255), thickness - 1)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                cap.release()
                cv2.destroyAllWindows()
                return
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Video playback complete\n")

# ================= MAIN LOOP =================
def main():
    print("\n" + "=" * 70)
    print("🎙️ VOICE TO SKELETON SIGN LANGUAGE CONVERTER")
    print("=" * 70)
    print("\nHow to use:")
    print("  1. Speak naturally into the microphone")
    print("  2. The matching skeleton video will play automatically")
    print("  3. Press 'q' or 'ESC' during playback to skip")
    print("  4. Press Ctrl+C to exit program")
    print("\n" + "=" * 70)
    
    try:
        while True:
            # Listen for speech
            text = listen_for_speech()
            
            if text:
                # Find matching sign
                match, score = find_best_match(text, available_signs)
                
                if match:
                    print(f"🎯 Best match: '{match}' (similarity: {score}%)")
                    video_path = available_signs[match]
                    play_sign_video(video_path, match)
                else:
                    print(f"❌ No matching sign found for: '{text}'")
                    print(f"   (Best similarity: {score}% - threshold: {SIMILARITY_THRESHOLD}%)")
                    print(f"\n   Available signs:")
                    for sign in sorted(available_signs.keys()):
                        print(f"      - {sign}")
            
            print("\n" + "-" * 70)
    
    except KeyboardInterrupt:
        print("\n\n👋 Exiting... Goodbye!")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()