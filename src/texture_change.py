import cv2
import numpy as np 
from skimage.metrics import structural_similarity as ssim 
def detect_surface_change(baseline_path, current_frame, surface_region): 
    baseline = cv2.imread(baseline_path) 
    sx, sy, sw, sh = surface_region 
    baseline_crop = cv2.cvtColor(baseline[sy:sy+sh, sx:sx+sw], cv2.COLOR_BGR2GRAY) 
    current_crop = cv2.cvtColor(current_frame[sy:sy+sh, sx:sx+sw], cv2.COLOR_BGR2GRAY) 
    current_crop = cv2.resize(current_crop, (baseline_crop.shape[1], baseline_crop.shape[0])) 
    score, diff = ssim(baseline_crop, current_crop, full=True) 
    diff = (diff * 255).astype("uint8") 
    # Threshold the diff to get a binary "changed surface" mask 
    _, mask = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV) 
    change_percent = float(np.sum(mask > 0) / mask.size * 100) 
    return { 
        "similarity_score": float(score),   # 1.0 = identical, lower = more changed 
        "change_percent": change_percent, 
        "change_mask": mask  
                     }                 # this is your "modified surface mask" for visualization