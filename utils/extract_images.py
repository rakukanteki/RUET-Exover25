import cv2
import os


video_path = "camera_recording/camera_recording.avi"
output_folder = "images"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(fps*5)
frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval == 0:
        cv2.imwrite(f"{output_folder}/frame_{frame_idx:05d}.jpg", frame)
        saved_idx += 1
    
    frame_idx += 1

cap.release()
print("Done!")