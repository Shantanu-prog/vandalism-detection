import cv2
def read_frames(video_path, sample_every_n_frames=5): 
    """Yields (frame_index, timestamp_seconds, frame) for sampled frames.""" 
    cap = cv2.VideoCapture(video_path) 
    frame_idx = 0 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
        if frame_idx % sample_every_n_frames == 0: 
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 
            yield frame_idx, timestamp, frame 
        frame_idx += 1 
    cap.release() 
if __name__ == "__main__": 
    for idx, ts, frame in read_frames("uploads/sample_surveillance.mp4"): 
        print(f"Frame {idx} at {ts:.2f}s, shape={frame.shape}") 