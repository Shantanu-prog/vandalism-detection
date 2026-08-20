import cv2
import numpy as np 
def annotate_frame(frame, surface_region, change_mask, wrist_position=None, is_alert=False): 
    annotated = frame.copy() 
    sx, sy, sw, sh = surface_region 
    # Overlay the changed-surface mask in red on the surface region 
    mask_colored = np.zeros_like(annotated[sy:sy+sh, sx:sx+sw]) 
    mask_colored[:, :, 2] = change_mask  # red channel 
    annotated[sy:sy+sh, sx:sx+sw] = cv2.addWeighted( 
        annotated[sy:sy+sh, sx:sx+sw], 0.7, mask_colored, 0.3, 0 
    )
    # Draw the surface region box 
    box_color = (0, 0, 255) if is_alert else (0, 255, 0) 
    cv2.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), box_color, 2) 
    # Mark the wrist/hand position if detected 
    if wrist_position: 
        cv2.circle(annotated, wrist_position, 8, (255, 0, 0), -1) 
    if is_alert: 
        cv2.putText(annotated, "SUSPICIOUS ACTIVITY", (sx, sy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
    return annotated