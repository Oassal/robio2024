from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def extract_masks_with_tracking(frame, frame_number, model=None):
    """Extract masks from a frame using tracking and return a dictionary of masks"""
    masks_dict = {}
    # print(f'frame shape : {frame.shape}')
    # Run tracking
    try:
        results = model.track(frame, persist=True, verbose = False)
    except:
        results = model.predict(frame)
    if results and len(results) > 0:
        # Process each detection
        for ci, c in enumerate(results[0]):
            if hasattr(c, 'masks') and hasattr(c.masks, 'xy'):
                label = c.names[c.boxes.cls.tolist().pop()]
                track_id = c.boxes.id.int().cpu().tolist().pop() if c.boxes.id is not None else ci
                b_mask = np.zeros(frame.shape[:2], np.uint8)
                
                # Create contour mask
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                
                # Store mask in dictionary with frame number and track ID as key
                key = label
                masks_dict[key] = b_mask
    
    return masks_dict if results else None

def process_video(video_path,model_path, save_dir="Masks" ):
    """Process video frame by frame and extract masks"""
    # Load model once
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    
    while cap.isOpened():
        # Read frame
        success, frame = cap.read()
        
        if not success:
            break
            
        # Process frame
        results= extract_masks_with_tracking(frame, frame_count, model)
        
        frame_count += 1
        
        # Optional: Display progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
            
        # Break if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# # Example usage
# video_path = "20240530-162620046.webm"
# process_video(video_path, save_dir="Masks", model_path=r"runs\segment\train13\weights\best.pt")