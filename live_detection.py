import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import time
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'sign_language_mlp_model.keras'
SCALER_PATH = 'mlp_scaler.pkl'
LABEL_ENCODER_PATH = 'mlp_label_encoder.pkl'
METADATA_PATH = 'mlp_metadata.pkl'

CONFIDENCE_THRESHOLD = 0.85
DETECTION_COOLDOWN = 2.0
VOTING_WINDOW = 12
MIN_DETECTION_FRAMES = 8

# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model_and_metadata():
    """Load trained MLP model and metadata"""
    print("Loading MLP model...")
    
    model = keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Model loaded successfully!")
    print(f"Classes: {metadata['classes']}")
    print(f"Model accuracy: {metadata['accuracy']:.2%}")
    
    return model, scaler, label_encoder, metadata

# ============================================================================
# MEDIAPIPE HAND TRACKING
# ============================================================================

class HandLandmarkExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame):
        """Extract hand landmarks and compute features"""
        # DON'T flip here - process the original frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        landmarks = np.zeros(126)
        hands_detected = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            hands_detected = True
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                                  results.multi_handedness):
                # Since we flip the display, swap left/right for consistency with training
                hand_label = handedness.classification[0].label
                
                coords = []
                for landmark in hand_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                # Place in correct position
                if hand_label == "Left":
                    landmarks[0:63] = coords
                else:
                    landmarks[63:126] = coords
        
        # Add derived features
        features = list(landmarks)
        
        left_hand_x = np.mean([landmarks[i] for i in range(0, 63, 3)])
        left_hand_y = np.mean([landmarks[i] for i in range(1, 63, 3)])
        right_hand_x = np.mean([landmarks[i] for i in range(63, 126, 3)])
        right_hand_y = np.mean([landmarks[i] for i in range(64, 126, 3)])
        hand_distance = np.sqrt((left_hand_x - right_hand_x)**2 + 
                               (left_hand_y - right_hand_y)**2)
        
        features.extend([left_hand_x, left_hand_y, 
                        right_hand_x, right_hand_y, 
                        hand_distance])
        
        return np.array(features), results, hands_detected
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def close(self):
        self.hands.close()

# ============================================================================
# MLP GESTURE DETECTOR
# ============================================================================

class MLPGestureDetector:
    """MLP-based gesture detector with voting"""
    
    def __init__(self, model, scaler, label_encoder, metadata):
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.metadata = metadata
        
        self.last_detection_time = 0
        self.detection_history = []
        self.recent_predictions = []
        self.last_detected_gesture = None
    
    def predict(self, features, hands_detected):
        """Predict gesture from features"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_detection_time < DETECTION_COOLDOWN:
            time_remaining = DETECTION_COOLDOWN - (current_time - self.last_detection_time)
            return {
                'gesture': None,
                'confidence': 0.0,
                'status': f'Cooldown: {time_remaining:.1f}s',
                'all_probabilities': {}
            }
        
        # Check hands
        if not hands_detected:
            self.recent_predictions.clear()
            return {
                'gesture': None,
                'confidence': 0.0,
                'status': 'No hands detected',
                'all_probabilities': {}
            }
        
        # Normalize and predict
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        probabilities = self.model.predict(features_scaled, verbose=0)[0]
        
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        gesture = self.label_encoder.classes_[predicted_class_idx]
        
        # Probability dictionary
        prob_dict = {
            self.label_encoder.classes_[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }
        
        # Add to voting if confident enough
        if confidence > 0.5:
            self.recent_predictions.append(gesture)
            if len(self.recent_predictions) > VOTING_WINDOW:
                self.recent_predictions.pop(0)
        
        # Voting
        if len(self.recent_predictions) >= MIN_DETECTION_FRAMES:
            vote_counts = Counter(self.recent_predictions)
            voted_gesture, vote_count = vote_counts.most_common(1)[0]
            vote_confidence = vote_count / len(self.recent_predictions)
            
            # Require high voting AND high model confidence
            if vote_confidence > CONFIDENCE_THRESHOLD and confidence > 0.75:
                detected_gesture = voted_gesture
                detected_confidence = vote_confidence
                
                self.last_detected_gesture = {
                    'gesture': detected_gesture,
                    'confidence': detected_confidence,
                    'time': current_time
                }
                
                self.last_detection_time = current_time
                self.detection_history.append({
                    'gesture': detected_gesture,
                    'confidence': detected_confidence,
                    'time': current_time
                })
                
                return {
                    'gesture': detected_gesture,
                    'confidence': detected_confidence,
                    'status': 'DETECTED!',
                    'all_probabilities': prob_dict
                }
        
        return {
            'gesture': gesture,
            'confidence': confidence,
            'status': 'Monitoring...',
            'all_probabilities': prob_dict
        }

# ============================================================================
# UI FUNCTIONS
# ============================================================================

def draw_info_panel(frame, prediction, fps, hands_detected, classes, last_detected):
    """Draw information panel"""
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    panel_height = 220 + (len(classes) * 25)
    cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    cv2.putText(frame, "MLP Sign Language Detector", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Detected phrase
    if last_detected:
        detected_text = f"Detected: {last_detected['gesture'].upper()}"
        cv2.putText(frame, detected_text, (20, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Detected: (waiting...)", (20, 75),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 150, 150), 2)
    
    # Hand status
    hand_status = "Hands: DETECTED" if hands_detected else "Hands: NOT DETECTED"
    hand_color = (0, 255, 0) if hands_detected else (0, 0, 255)
    cv2.putText(frame, hand_status, (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
    
    if prediction:
        status = prediction['status']
        color = (0, 255, 0) if prediction['gesture'] and 'DETECTED' in status else (100, 100, 255)
        cv2.putText(frame, f"Status: {status}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Probabilities
        y_pos = 170
        cv2.putText(frame, "Class Probabilities:", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, class_name in enumerate(classes):
            y_pos += 25
            prob = prediction['all_probabilities'].get(class_name, 0.0)
            bar_width = int(300 * prob)
            
            cv2.rectangle(frame, (150, y_pos - 15), (450, y_pos), (50, 50, 50), -1)
            bar_color = (0, 255, 0) if prob > 0.6 else (100, 100, 255)
            cv2.rectangle(frame, (150, y_pos - 15), (150 + bar_width, y_pos), bar_color, -1)
            
            cv2.putText(frame, f"{class_name}:", (20, y_pos - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.putText(frame, f"{prob:.1%}", (460, y_pos - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.putText(frame, "Press 'Q' to quit | 'C' to clear history", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def draw_detection_history(frame, detection_history):
    """Draw recent detections"""
    h, w = frame.shape[:2]
    
    if detection_history:
        recent = detection_history[-5:]
        
        y_start = h - 180
        cv2.putText(frame, "Recent Detections:", (w - 350, y_start - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, detection in enumerate(recent):
            y_pos = y_start + (i * 30)
            text = f"• {detection['gesture']} ({detection['confidence']:.1%})"
            cv2.putText(frame, text, (w - 350, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    
    return frame

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main detection loop"""
    print("=" * 70)
    print("MLP SIGN LANGUAGE DETECTION")
    print("=" * 70)
    
    try:
        model, scaler, label_encoder, metadata = load_model_and_metadata()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrain the MLP model first!")
        print("Run: python training_mlp.py")
        return
    
    classes = metadata['classes']
    
    print("\nInitializing camera and hand tracking...")
    hand_extractor = HandLandmarkExtractor()
    gesture_detector = MLPGestureDetector(model, scaler, label_encoder, metadata)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nCamera started!")
    print(f"Detecting gestures: {classes}")
    print("\nControls:")
    print("  - Press 'Q' to quit")
    print("  - Press 'C' to clear history\n")
    
    fps_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            features, results, hands_detected = hand_extractor.extract_landmarks(frame)
            frame = hand_extractor.draw_landmarks(frame, results)
            
            prediction = gesture_detector.predict(features, hands_detected)
            
            current_time = time.time()
            fps = 1 / (current_time - fps_time + 1e-6)
            fps_time = current_time
            
            last_detected = gesture_detector.last_detected_gesture
            
            frame = draw_info_panel(frame, prediction, fps, hands_detected, classes, last_detected)
            frame = draw_detection_history(frame, gesture_detector.detection_history)
            
            cv2.imshow('MLP Sign Language Detector', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                gesture_detector.detection_history.clear()
                print("Detection history cleared")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hand_extractor.close()
        
        print("\n" + "=" * 70)
        print("SESSION SUMMARY")
        print("=" * 70)
        print(f"Total detections: {len(gesture_detector.detection_history)}")
        if gesture_detector.detection_history:
            print("\nDetected gestures:")
            for i, det in enumerate(gesture_detector.detection_history, 1):
                print(f"  {i}. '{det['gesture']}' - {det['confidence']:.1%} confidence")
        print("\nGoodbye!")

if __name__ == "__main__":
    main()