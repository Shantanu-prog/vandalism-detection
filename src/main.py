import os
import cv2
import tensorflow as tf
from src.video_input import read_frames
from src.motion_features import get_upper_limb_features, track_proximity_duration
from src.texture_change import detect_surface_change
from src.action_model import predict_frame, MODEL_PATH
from src.alerts import check_alert
from src.storage import init_db, save_event
from src.report import export_csv
from src.visualize import annotate_frame

VIDEO_PATH = "uploads/sample_surveillance.mp4"
BASELINE_PATH = "data/baseline_frame.jpg"
LOCATION = "Monument-Gate-1"
SURFACE_REGION = (50, 50, 400, 300)   # (x, y, w, h) — adjust to where the wall/monument actually is in your footage
PROXIMITY_WINDOW = 16                  # how many recent sampled frames to check for sustained hand proximity

def run_pipeline():
    os.makedirs("outputs", exist_ok=True)
    init_db()

    model = tf.keras.models.load_model(MODEL_PATH)
    all_features = []
    frame_count = 0
    alert_count = 0

    for idx, ts, frame in read_frames(VIDEO_PATH):
        # Upper-limb motion + surface proximity (Core Requirement 2)
        motion = get_upper_limb_features(frame, SURFACE_REGION)
        all_features.append(motion)
        proximity_duration = track_proximity_duration(all_features[-PROXIMITY_WINDOW:])

        # Deep learning action classification on this frame (Core Requirement 3)
        action_result = predict_frame(frame, model=model)

        # Texture/structural change vs. baseline (Core Requirement 4)
        change_result = detect_surface_change(BASELINE_PATH, frame, SURFACE_REGION)

        # Configurable alert check (Core Requirement 5)
        triggered, message = check_alert(action_result, change_result, proximity_duration)

        # --- Expected Output 1: visualization with suspicious regions + surface mask ---
        annotated = annotate_frame(
            frame, SURFACE_REGION, change_result["change_mask"],
            wrist_position=motion.get("wrist_position"), is_alert=triggered
        )
        cv2.imwrite(f"outputs/annotated_frame_{idx}.jpg", annotated)

        # --- Expected Output 2: alert when confidence crosses threshold ---
        if triggered:
            alert_count += 1
            print(message)

        # Log every analyzed frame as an event (feeds Expected Output 3)
        save_event(LOCATION, action_result["label"], action_result["confidence"], triggered)
        frame_count += 1

    # --- Expected Output 3: CSV report ---
    csv_path = export_csv()

    print(f"\nPipeline complete.")
    print(f"Frames analyzed: {frame_count}")
    print(f"Alerts triggered: {alert_count}")
    print(f"Annotated frames saved to: outputs/annotated_frame_*.jpg")
    print(f"CSV report saved to: {csv_path}")

if __name__ == "__main__":
    run_pipeline()