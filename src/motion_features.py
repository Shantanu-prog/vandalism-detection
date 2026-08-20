import mediapipe as mp 
import numpy as np 
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) 
def get_upper_limb_features(frame, surface_region): 
    """ 
    surface_region: (x, y, w, h) box marking the wall/monument area in the frame. 
    Returns dict with wrist positions and proximity-to-surface info for this frame. 
    """ 
    rgb = frame[:, :, ::-1] 
    results = pose.process(rgb) 
    if not results.pose_landmarks: 
        return {"hand_detected": False, "distance_to_surface": None} 
    h, w, _ = frame.shape 
    landmarks = results.pose_landmarks.landmark 
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST] 
    wrist_x, wrist_y = int(right_wrist.x * w), int(right_wrist.y * h) 
    sx, sy, sw, sh = surface_region 
    surface_center = (sx + sw / 2, sy + sh / 2) 
    distance = float(np.hypot(wrist_x - surface_center[0], wrist_y - surface_center[1])) 
    return { 
        "hand_detected": True, 
        "wrist_position": (wrist_x, wrist_y), 
        "distance_to_surface": distance 
    }
def track_proximity_duration(frames_features, proximity_threshold=80): 
    """Counts consecutive frames where the hand stayed close to the surface.""" 
    duration = 0 
    for f in frames_features: 
        if f["hand_detected"] and f["distance_to_surface"] is not None and f["distance_to_surface"] < proximity_threshold: 
            duration += 1 
    return duration

