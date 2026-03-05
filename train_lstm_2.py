import numpy as np
import os
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_PATH = "keypoints_data2"
labels = os.listdir(DATA_PATH)

label_map = {label: num for num, label in enumerate(labels)}
pickle.dump(label_map, open("label_map.pkl", "wb"))

X, y = [], []

for label in labels:
    label_path = os.path.join(DATA_PATH, label)

    # Find how many videos there are by checking folder 0
    first_frame_folder = os.path.join(label_path, "0")
    num_videos = len(os.listdir(first_frame_folder))

    for video_idx in range(num_videos):
        sequence = []

        for frame_idx in range(30):  # 30 frame folders
            frame_folder = os.path.join(label_path, str(frame_idx))
            npy_file = os.path.join(frame_folder, f"{video_idx}.npy")

            if os.path.exists(npy_file):
                keypoints = np.load(npy_file)
                sequence.append(keypoints)
            else:
                # If a frame is missing, fill with zeros
                sequence.append(np.zeros(126))

        sequence = np.array(sequence)  # shape: (30, 126)

        if sequence.shape == (30, 126):
            X.append(sequence)
            y.append(label_map[label])
        else:
            print(f"⚠ Skipped video {video_idx} for '{label}' — unexpected shape {sequence.shape}")

print(f"\n✅ Loaded {len(X)} sequences across {len(labels)} labels")

X = np.array(X)  # shape: (num_samples, 30, 126)
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
model.save("model.h5")

print("\n✅ Training complete. Model saved as model.h5")