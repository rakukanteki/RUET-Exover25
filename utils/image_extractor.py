import cv2
import os

# List of video paths
video_paths = [
    r"D:\\Competition\\NASARoverRally\\RUET-Exover25\\camera_recording\\camera_recording.avi",
    r"D:\\Competition\\NASARoverRally\\RUET-Exover25\\camera_recording\\camera_recording2.avi"
]

# Output folder
output_dir = r"D:\Competition\NASARoverRally\RUET-Exover25\extracted_frames"
os.makedirs(output_dir, exist_ok=True)

# Extract 1 frame every 2 seconds
FRAME_INTERVAL_SECONDS = 2

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * FRAME_INTERVAL_SECONDS)

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            save_path = os.path.join(
                output_dir, f"{video_name}_frame_{saved_count:05d}.jpg"
            )
            cv2.imwrite(save_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Finished extracting frames from {video_path}")

print("Done!")
