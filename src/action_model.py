import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os
import random
import re

# Configuration
IMAGE_SIZE = (64, 64)  # Resize all images to 64x64 pixels
DATA_DIR = "data/training_clips"
MODEL_PATH = "outputs/action_model.h5"
MAX_IMAGES_PER_CLASS = 3000  # cap to avoid loading huge frame-extracted datasets into RAM at once

def build_model():
    """
    Builds a 2D CNN model for image classification.
    Input: (64, 64, 3)
    Output: Probability of [normal, suspicious]

    Includes light data augmentation as the first layers — this only
    runs during training, never during prediction. It makes it harder
    for the model to memorize a specific video's exact lighting/framing,
    which is what caused the overfitting seen in earlier runs.
    """
    model = Sequential([
        tf.keras.layers.RandomFlip("horizontal", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomZoom(0.1),

        # Block 1
        Conv2D(32, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        # Block 2
        Conv2D(64, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        # Block 3
        Conv2D(128, (3, 3), activation="relu"),
        MaxPool2D((2, 2)),

        # Classifier
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),  # Prevents overfitting
        Dense(2, activation="softmax")  # 2 classes: normal, suspicious
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_subfolder_path(base_dir):
    """
    Auto-detects the single subfolder inside base_dir instead of hardcoding
    a name like 'NormalVideos'. This avoids breakage from naming variations
    (e.g. 'NormalVideo' vs 'NormalVideos') between what you actually created
    and what any code assumes.
    """
    if not os.path.exists(base_dir):
        return None
    subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    if not subfolders:
        return None
    if len(subfolders) > 1:
        print(f"⚠️ Multiple subfolders found in {base_dir}: {subfolders} — using '{subfolders[0]}'")
    return os.path.join(base_dir, subfolders[0])

def load_dataset_video_split(data_dir, max_frames_per_class=3000, val_video_fraction=0.2):
    """
    Splits by VIDEO, not by frame, so validation is a genuine test of
    generalization to unseen videos instead of near-duplicate frames
    from the same video leaking into both train and validation.

    Also samples frames evenly per video (not pooled random) so one
    long video can't dominate the training set and teach the model
    that video's specific visual signature instead of general vandalism cues.
    """
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    X_train, y_train, X_val, y_val = [], [], [], []

    for label, folder in enumerate(["normal", "suspicious"]):
        base_dir = os.path.join(data_dir, folder)
        folder_path = get_subfolder_path(base_dir)
        if folder_path is None:
            print(f"⚠️ No subfolder found inside: {base_dir}")
            continue

        # Group frames by their source video
        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_exts)]
        videos = {}
        for f in all_files:
            video_id = re.sub(r'_[0-9]+\.(png|jpg|jpeg|bmp|webp)$', '', f, flags=re.IGNORECASE)
            videos.setdefault(video_id, []).append(f)

        video_ids = list(videos.keys())
        random.shuffle(video_ids)
        n_val_videos = max(1, int(len(video_ids) * val_video_fraction))
        val_video_ids = set(video_ids[:n_val_videos])
        train_video_ids = video_ids[n_val_videos:]

        frames_per_video = max(1, max_frames_per_class // max(len(train_video_ids), 1))

        print(f"{folder}: {len(video_ids)} videos total "
              f"({len(train_video_ids)} train / {len(val_video_ids)} val), "
              f"~{frames_per_video} frames/video from train videos")

        for vid in train_video_ids:
            frame_files = videos[vid]
            random.shuffle(frame_files)
            for fname in frame_files[:frames_per_video]:
                img = cv2.imread(os.path.join(folder_path, fname))
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE) / 255.0
                    X_train.append(img)
                    y_train.append(label)

        for vid in val_video_ids:
            frame_files = videos[vid]
            random.shuffle(frame_files)
            for fname in frame_files[:frames_per_video]:
                img = cv2.imread(os.path.join(folder_path, fname))
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE) / 255.0
                    X_val.append(img)
                    y_val.append(label)

    return (np.array(X_train), np.array(y_train)), (np.array(X_val), np.array(y_val))

def load_dataset(data_dir, max_images_per_class=MAX_IMAGES_PER_CLASS):
    X, y = [], []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for label, folder in enumerate(["normal", "suspicious"]):
        base_dir = os.path.join(data_dir, folder)
        folder_path = get_subfolder_path(base_dir)

        if folder_path is None:
            print(f"⚠️ No subfolder found inside: {base_dir}")
            continue

        print(f"Loading images from {folder_path}...")

        filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(image_exts)]

        # Shuffle before capping so we don't just take the first N alphabetically
        # (which could bias toward frames from only the first video or two)
        random.shuffle(filenames)
        if max_images_per_class is not None and len(filenames) > max_images_per_class:
            print(f"   Found {len(filenames)} images — capping to {max_images_per_class} for memory safety")
            filenames = filenames[:max_images_per_class]

        count = 0
        for filename in filenames:
            filepath = os.path.join(folder_path, filename)
            img = cv2.imread(filepath)

            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                img = img / 255.0
                X.append(img)
                y.append(label)
                count += 1

        print(f"✅ Loaded {count} images from {folder}")

    if len(X) == 0:
        raise ValueError("No images found! Check paths.")

    return np.array(X), np.array(y)

def train():
    print("Starting training process...")
    try:
        (X_train, y_train), (X_val, y_val) = load_dataset_video_split(DATA_DIR, max_frames_per_class=3000)
        print(f"Train shape: {X_train.shape} | Val shape: {X_val.shape}")
        print(f"Train balance -> normal: {np.sum(y_train == 0)}, suspicious: {np.sum(y_train == 1)}")
        print(f"Val balance   -> normal: {np.sum(y_val == 0)}, suspicious: {np.sum(y_val == 1)}")

        model = build_model()
        model.summary()

        # Now validating on entirely separate VIDEOS, not just separate frames —
        # this val_accuracy is a trustworthy signal of real generalization.
        history = model.fit(
            X_train, y_train, epochs=15, batch_size=32,
            validation_data=(X_val, y_val)
        )

        # Save the model
        os.makedirs("outputs", exist_ok=True)
        model.save(MODEL_PATH)
        print(f"✅ Model successfully saved to {MODEL_PATH}")

    except Exception as e:
        print(f"❌ Error during training: {e}")

def predict_single_image(image_path):
    """Test the model on a single image file."""
    if not os.path.exists(MODEL_PATH):
        print("❌ No trained model found. Run train() first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    img = cv2.imread(image_path)

    if img is None:
        print("❌ Could not read image.")
        return

    # Preprocess
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)
    label_idx = np.argmax(prediction[0])
    confidence = prediction[0][label_idx]
    label_name = "suspicious" if label_idx == 1 else "normal"

    print(f"\n📷 Image: {image_path}")
    print(f"🔍 Prediction: {label_name.upper()}")
    print(f"📊 Confidence: {confidence:.2%}")

if __name__ == "__main__":
    # Run training
    train()

    # Optional: Test on a specific image after training
    # predict_single_image("data/test_clips/normal/test_image.jpg")