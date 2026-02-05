import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ============================================================================
# MLP APPROACH FOR SIGN LANGUAGE DETECTION
# ============================================================================

def load_and_prepare_data(csv_files):
    """Load multiple CSV files and prepare data"""
    print("=" * 70)
    print("LOADING DATA FROM MULTIPLE FILES")
    print("=" * 70)
    
    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        print(f"\nLoading: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"  - Shape: {df.shape}")
        print(f"  - Phrases: {df['phrase'].unique()}")
        print(f"  - Frames: {len(df)}")
        dataframes.append(df)
    
    # Combine all dataframes
    df = pd.concat(dataframes, ignore_index=True)
    
    print("\n" + "=" * 70)
    print("COMBINED DATASET")
    print("=" * 70)
    print(f"Total shape: {df.shape}")
    print(f"Total frames: {len(df)}")
    print(f"\nUnique phrases: {df['phrase'].nunique()}")
    print(f"All phrases: {df['phrase'].unique()}")
    
    # Show distribution of phrases
    print("\nPhrase distribution:")
    for phrase in df['phrase'].unique():
        count = len(df[df['phrase'] == phrase])
        percentage = (count / len(df)) * 100
        print(f"  - '{phrase}': {count} frames ({percentage:.1f}%)")
    
    # Extract coordinate columns
    coord_columns = [col for col in df.columns if any(
        keyword in col for keyword in ['_x', '_y', '_z']
    )]
    
    # Get features (coordinates) and labels (phrases)
    X = df[coord_columns].values
    y = df['phrase'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Each frame has {X.shape[1]} coordinate values")
    
    # Add derived features
    hand_features = []
    for i in range(len(X)):
        frame_features = list(X[i])
        
        # Left hand center
        left_hand_x = np.mean([X[i][j] for j in range(0, 63, 3)])
        left_hand_y = np.mean([X[i][j] for j in range(1, 63, 3)])
        
        # Right hand center
        right_hand_x = np.mean([X[i][j] for j in range(63, 126, 3)])
        right_hand_y = np.mean([X[i][j] for j in range(64, 126, 3)])
        
        # Distance between hands
        hand_distance = np.sqrt((left_hand_x - right_hand_x)**2 + 
                               (left_hand_y - right_hand_y)**2)
        
        frame_features.extend([left_hand_x, left_hand_y, 
                              right_hand_x, right_hand_y, 
                              hand_distance])
        
        hand_features.append(frame_features)
    
    X = np.array(hand_features)
    
    print(f"Enhanced features shape: {X.shape}")
    
    return X, y

def build_mlp_model(input_dim, num_classes):
    """Build MLP neural network - Enhanced version"""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer - increased size
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second hidden layer - increased size
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third hidden layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Fourth hidden layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Fifth hidden layer
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ], name='mlp_classifier')
    
    return model

def train_mlp_model(X, y, epochs=150, batch_size=16):  # Increased epochs, smaller batch
    """Train MLP classifier"""
    print("\n" + "=" * 70)
    print("TRAINING MLP NEURAL NETWORK")
    print("=" * 70)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = keras.utils.to_categorical(y_encoded)
    
    print(f"\nClasses: {label_encoder.classes_}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, 
        stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Normalize data
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Build model
    num_classes = len(label_encoder.classes_)
    model = build_mlp_model(input_dim=X_train.shape[1], num_classes=num_classes)
    
    # Compile model with higher learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),  # Increased from 0.001
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks with adjusted patience
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased from 15 - train longer
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,  # Increased from 7
        min_lr=0.00001
    )
    
    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Training accuracy
    train_predictions = model.predict(X_train, verbose=0)
    train_pred_classes = np.argmax(train_predictions, axis=1)
    train_true_classes = np.argmax(y_train, axis=1)
    train_accuracy = accuracy_score(train_true_classes, train_pred_classes)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    
    # Testing accuracy
    test_predictions = model.predict(X_test, verbose=0)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = np.argmax(y_test, axis=1)
    test_accuracy = accuracy_score(test_true_classes, test_pred_classes)
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(classification_report(test_true_classes, test_pred_classes, 
                                target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(test_true_classes, test_pred_classes)
    plot_confusion_matrix(cm, label_encoder.classes_)
    
    # Plot training history
    plot_training_history(history)
    
    return model, scaler, label_encoder, test_accuracy

def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_mlp.png', dpi=150)
    print("\nConfusion matrix saved as 'confusion_matrix_mlp.png'")
    plt.close()

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_mlp.png', dpi=150)
    print("Training history saved as 'training_history_mlp.png'")
    plt.close()

def save_model(model, scaler, label_encoder, accuracy):
    """Save trained model"""
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    try:
        # Save Keras model
        model.save('sign_language_mlp_model.keras')
        print("✓ Model saved as 'sign_language_mlp_model.keras'")
        
        # Save scaler
        with open('mlp_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("✓ Scaler saved as 'mlp_scaler.pkl'")
        
        # Save label encoder
        with open('mlp_label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print("✓ Label encoder saved as 'mlp_label_encoder.pkl'")
        
        # Save metadata
        metadata = {
            'model_type': 'MLP',
            'classes': list(label_encoder.classes_),
            'num_classes': len(label_encoder.classes_),
            'accuracy': accuracy
        }
        
        with open('mlp_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print("✓ Metadata saved as 'mlp_metadata.pkl'")
        
        # Verify files
        import os
        if os.path.exists('sign_language_mlp_model.keras'):
            print("\n✓ Verified: sign_language_mlp_model.keras exists")
        if os.path.exists('mlp_scaler.pkl'):
            print("✓ Verified: mlp_scaler.pkl exists")
        if os.path.exists('mlp_label_encoder.pkl'):
            print("✓ Verified: mlp_label_encoder.pkl exists")
        if os.path.exists('mlp_metadata.pkl'):
            print("✓ Verified: mlp_metadata.pkl exists")
            
    except Exception as e:
        print(f"\n❌ ERROR saving files: {e}")
        raise

def main():
    """Main training pipeline"""
    
    # LIST YOUR CSV FILES HERE
    CSV_FILES = [
        "good_morning_doctor.csv",
        "what_brings_you_here.csv"
    ]
    
    print("\n" + "=" * 70)
    print("MLP SIGN LANGUAGE DETECTION")
    print("Neural Network Approach")
    print("=" * 70)
    print(f"\nTraining on {len(CSV_FILES)} gesture files:")
    for i, csv_file in enumerate(CSV_FILES, 1):
        print(f"  {i}. {csv_file}")
    
    # Load data
    X, y = load_and_prepare_data(CSV_FILES)
    
    # Train model with updated parameters
    model, scaler, label_encoder, accuracy = train_mlp_model(X, y, epochs=150, batch_size=16)
    
    # Save model
    save_model(model, scaler, label_encoder, accuracy)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Test Accuracy: {accuracy:.2%}")
    print(f"Classes trained: {list(label_encoder.classes_)}")
    print("\nFiles created:")
    print("  - sign_language_mlp_model.keras")
    print("  - mlp_scaler.pkl")
    print("  - mlp_label_encoder.pkl")
    print("  - mlp_metadata.pkl")
    print("  - confusion_matrix_mlp.png")
    print("  - training_history_mlp.png")
    print("\nNext step: Run live detection")
    print("  python live_detection_mlp.py")

if __name__ == "__main__":
    main()