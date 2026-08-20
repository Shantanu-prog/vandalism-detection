"""
Evaluate the trained action_model on a held-out test set.

Usage:
    python -m src.test_model

Expects a folder structure separate from your training data, e.g.:
    data/test_clips/normal/*.jpg
    data/test_clips/suspicious/*.jpg

IMPORTANT: these should be frames from videos NOT used in training
(a different subset of your UCF-Crime videos), not just leftover
frames from the same videos, or your accuracy number will be
misleadingly high (see explanation in chat).
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.action_model import IMAGE_SIZE, MODEL_PATH, get_subfolder_path

TEST_DIR = "data/test_clips"

def load_test_set(test_dir):
    """
    Expects the same nested structure as your training data:
    data/test_clips/normal/NormalVideos/*.jpg
    data/test_clips/suspicious/Vandalism/*.jpg
    (auto-detects the subfolder name, same as load_dataset() in action_model.py)
    """
    X, y, paths = [], [], []
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for label, folder in enumerate(["normal", "suspicious"]):
        base_dir = os.path.join(test_dir, folder)
        folder_path = get_subfolder_path(base_dir)
        if folder_path is None:
            print(f"⚠️ No subfolder found inside: {base_dir}")
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_exts):
                filepath = os.path.join(folder_path, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE) / 255.0
                    X.append(img)
                    y.append(label)
                    paths.append(filepath)

    return np.array(X), np.array(y), paths

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print("❌ No trained model found. Run train() first.")
        return

    X_test, y_test, paths = load_test_set(TEST_DIR)
    if len(X_test) == 0:
        print(f"❌ No test images found in {TEST_DIR}. Create data/test_clips/normal and data/test_clips/suspicious with held-out frames.")
        return

    print(f"Loaded {len(X_test)} test images "
          f"(normal: {np.sum(y_test == 0)}, suspicious: {np.sum(y_test == 1)})")

    model = tf.keras.models.load_model(MODEL_PATH)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print("\n=== Results ===")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {precision:.2%}  (of predicted 'suspicious', how many were correct)")
    print(f"Recall:    {recall:.2%}  (of actual 'suspicious', how many were caught)")
    print(f"F1 score:  {f1:.2%}")

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print("                predicted_normal  predicted_suspicious")
    print(f"actual_normal        {cm[0][0]:>6}              {cm[0][1]:>6}")
    print(f"actual_suspicious    {cm[1][0]:>6}              {cm[1][1]:>6}")

    print("\n=== Full report ===")
    print(classification_report(y_test, y_pred, target_names=["normal", "suspicious"]))

    # Show the model's worst mistakes — useful for spotting bad training data
    wrong = np.where(y_pred != y_test)[0]
    if len(wrong) > 0:
        print(f"\n{len(wrong)} misclassified images. Worst 5 (highest confidence, wrong answer):")
        wrong_sorted = sorted(wrong, key=lambda i: -confidences[i])[:5]
        for i in wrong_sorted:
            true_label = ["normal", "suspicious"][y_test[i]]
            pred_label = ["normal", "suspicious"][y_pred[i]]
            print(f"  {paths[i]} — true: {true_label}, predicted: {pred_label} ({confidences[i]:.2%} confident)")

    # --- Threshold sweep ---
    # argmax uses an implicit threshold of 0.5, which is a bad choice when
    # classes are this imbalanced. This checks a range of thresholds on the
    # raw "suspicious" probability to find one with a better precision/recall
    # tradeoff for your use case.
    print("\n=== Threshold sweep (raise this if you're getting too many false alarms) ===")
    suspicious_probs = predictions[:, 1]
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'alerts':>8}")
    best_f1, best_threshold = 0, 0.5
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        y_pred_thresh = (suspicious_probs >= threshold).astype(int)
        if y_pred_thresh.sum() == 0:
            p, r, f = 0, 0, 0
        else:
            p, r, f, _ = precision_recall_fscore_support(y_test, y_pred_thresh, average="binary", zero_division=0)
        print(f"{threshold:>10.2f} {p:>10.2%} {r:>10.2%} {f:>10.2%} {int(y_pred_thresh.sum()):>8}")
        if f > best_f1:
            best_f1, best_threshold = f, threshold
    print(f"\nBest F1 at threshold={best_threshold} (F1={best_f1:.2%}). "
          f"Use this threshold instead of argmax in your alert logic.")

if __name__ == "__main__":
    evaluate()