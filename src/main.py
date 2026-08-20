import cv2
import tensorflow as tf 
from src.video_input import read_frames 
from src.motion_features import get_upper_limb_features, track_proximity_duration 
from src.texture_change import detect_surface_change 
from src.action_model import predict_action, CLIP_LENGTH 
from src.alerts import check_alert 
from src.storage import init_db, save_event 
from src.report import export_csv 
from src.visualize import annotate_frame 
VIDEO_PATH = "uploads/sample_surveillance.mp4" 
BASELINE_PATH = "data/baseline_frame.jpg" 
LOCATION = "Monument-Gate-1" 
SURFACE_REGION = (50, 50, 400, 300)   # adjust to your footage 
def run_pipeline(): 
    init_db() 
    model = tf.keras.models.load_model("outputs/action_model.h5") 
    clip_buffer = [] 
    all_features = [] 
    for idx, ts, frame in read_frames(VIDEO_PATH): 
        clip_buffer.append(frame) 
        motion = get_upper_limb_features(frame, SURFACE_REGION) 
        all_features.append(motion) 
        if len(clip_buffer) < CLIP_LENGTH: 
            continue 
        # Run analysis every time we have a full clip window 
        action_result = predict_action(clip_buffer[-CLIP_LENGTH:], model=model) 
        change_result = detect_surface_change(BASELINE_PATH, frame, SURFACE_REGION) 
        proximity_duration = track_proximity_duration(all_features[-CLIP_LENGTH:]) 
        triggered, message = check_alert(action_result, change_result, proximity_duration) 
        # 1. Visualization output 
        annotated = annotate_frame( 
            frame, SURFACE_REGION, change_result["change_mask"], 
            wrist_position=motion.get("wrist_position"), is_alert=triggered 
        ) 
        cv2.imwrite(f"outputs/annotated_frame_{idx}.jpg", annotated) 
        # 2. Alert output 
        if triggered: 
            print(message) 
        # Log every analyzed frame as an event 
        save_event(LOCATION, action_result["label"], action_result["confidence"], triggered) 
    # 3. CSV report output 
    csv_path = export_csv() 
    print(f"Pipeline complete. CSV report at {csv_path}") 
if __name__ == "__main__": 
    run_pipeline() 