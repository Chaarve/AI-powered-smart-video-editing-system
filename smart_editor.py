"""
smart_editor.py -- Step 7: Video Understanding & Smart Editing

Adds AI-powered content-aware video editing features:
  - Face Detection & Blurring using MediaPipe
  - Object Detection & Tracking (Bounding boxes) using YOLOv8

Requires:
  pip install ultralytics mediapipe
"""

import os
import cv2
import numpy as np
from moviepy import VideoFileClip


class SmartVideoEditor:
    """Smart editor that understands video content (faces, objects)."""

    def __init__(self, input_path: str):
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")
        
        self.input_path = input_path
        self.clip = VideoFileClip(input_path)
        print(f"[SmartEditor] Loaded: {input_path}")
        print(f"  Duration : {self.clip.duration:.2f}s")
        print(f"  FPS      : {self.clip.fps}")

    # ================================================================== #
    #  1. Face Blurring (MediaPipe)
    # ================================================================== #
    def blur_faces(self, blur_intensity: int = 15):
        """
        Detect faces in the video and apply a Gaussian blur over them.
        Uses OpenCV's built-in Haar Cascade classifier.
        """
        print(f"[SmartEditor] Initializing OpenCV Face Detection (Haar)...")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        def _blur_faces_frame(get_frame, t):
            frame = get_frame(t)
            
            # Haar cascades work best on grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            output_frame = frame.copy()
            h, w, _ = frame.shape
            
            for (x, y, bw, bh) in faces:
                # Expand bounding box slightly for better coverage
                margin_x = int(bw * 0.15)
                margin_y = int(bh * 0.15)
                
                nx = max(0, x - margin_x)
                ny = max(0, y - margin_y)
                nxw = min(w, x + bw + margin_x * 2)
                nyh = min(h, y + bh + margin_y * 2)
                
                # Extract Region of Interest (ROI) and blur
                roi = output_frame[ny:nyh, nx:nxw]
                # kernel must be odd
                k_size = blur_intensity
                if k_size % 2 == 0:
                    k_size += 1
                blurred_roi = cv2.GaussianBlur(roi, (k_size, k_size), 30)
                output_frame[ny:nyh, nx:nxw] = blurred_roi
                        
            return output_frame

        self.clip = self.clip.transform(_blur_faces_frame)
        print(f"[SmartEditor] Applied Face Blurring (intensity={blur_intensity})")
        return self

    # ================================================================== #
    #  2. Object Detection & Tracking (YOLOv8)
    # ================================================================== #
    def highlight_object(self, object_name: str = "person"):
        """
        Draw a bounding box around specific objects detected by YOLOv8.
        Supports 80 standard COCO classes (person, car, dog, cat, etc.)
        """
        from ultralytics import YOLO
        print(f"[SmartEditor] Initializing YOLOv8 Object Detection...")
        
        # Load the nano model for speed (downloads automatically if missing)
        model = YOLO("yolov8n.pt") 

        # Map string name to YOLO class ID
        class_id = None
        for k, v in model.names.items():
            if v.lower() == object_name.lower():
                class_id = k
                break
                
        if class_id is None:
            print(f"  [Error] '{object_name}' is not recognized by standard YOLOv8.")
            print(f"          Supported: {list(model.names.values())[:10]}...")
            return self

        def _track_frame(get_frame, t):
            frame = get_frame(t)
            
            # Predict only the target class to speed up Post-processing
            results = model.predict(frame, classes=[class_id], verbose=False)
            output_frame = frame.copy()
            
            # Draw bounding boxes
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Draw rectangle and label
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"{object_name} {conf:.2f}"
                    cv2.putText(
                        output_frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    
            return output_frame

        self.clip = self.clip.transform(_track_frame)
        print(f"[SmartEditor] Applied Object Highlighting for '{object_name}'")
        return self

    # ================================================================== #
    #  Export & Cleanup
    # ================================================================== #
    def export(self, output_path: str, fps: int = None):
        if fps is None:
            fps = self.clip.fps
            
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self.clip.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )
        print(f"[SmartEditor] Exported -> {output_path}")

    def close(self):
        self.clip.close()
        print("[SmartEditor] Resources released.")


# ====================================================================== #
#  Quick Test
# ====================================================================== #
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  SmartVideoEditor -- Step 7 Quick Test")
    print("=" * 60)

    INPUT_VIDEO = "input/sample.mp4"
    OUTPUT_DIR = "output"

    if not os.path.isfile(INPUT_VIDEO):
        print(f"\nNo test video found at '{INPUT_VIDEO}'.")
        print("   Place a short .mp4 in the input/ folder and re-run.")
        sys.exit(1)

    # Note: ML models can be slow to run per-frame on CPU. 
    # To test, we'll clip the video to the first 3 seconds.
    print("\n--- Test 1: Content-Aware Face Blurring ---")
    editor = SmartVideoEditor(INPUT_VIDEO)
    editor.clip = editor.clip.subclipped(0, min(3.0, editor.clip.duration))
    editor.blur_faces(blur_intensity=25)
    editor.export(os.path.join(OUTPUT_DIR, "test_face_blur.mp4"))
    editor.close()

    print("\n--- Test 2: YOLO Object Tracking (Person) ---")
    editor2 = SmartVideoEditor(INPUT_VIDEO)
    editor2.clip = editor2.clip.subclipped(0, min(3.0, editor2.clip.duration))
    editor2.highlight_object(object_name="person")
    editor2.export(os.path.join(OUTPUT_DIR, "test_object_tracking.mp4"))
    editor2.close()

    print("\n✅ All SmartEditor tests complete. Check output/ directory.")
