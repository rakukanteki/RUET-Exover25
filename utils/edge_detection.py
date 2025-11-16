import cv2
import numpy as np

video_path = r"D:\\Competition\\NASARoverRally\\RUET-Exover25\\camera_recording\\camera_recording.avi"

# Parameters
alpha = 1.8  # Contrast
beta = 80    # Brightness
kernel = 5   # Gaussian blur kernel size
roi_width_ratio = 0.8
roi_height_ratio = 0.3
vertical_offset_ratio = 0.1  # Shift ROI downward by 20% of ROI height

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Increase contrast and brightness
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Grayscale + blur
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    # Canny edges
    canny = cv2.Canny(blur, 50, 150)
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    # ROI rectangle slightly below center
    h, w = enhanced.shape[:2]
    roi_w = int(w * roi_width_ratio)
    roi_h = int(h * roi_height_ratio)
    x1 = w//2 - roi_w//2
    y1 = h//2 - roi_h//2 + int(roi_h * vertical_offset_ratio)
    y2 = y1 + roi_h
    x2 = x1 + roi_w

    # Draw rectangle
    roi_img = enhanced.copy()
    cv2.rectangle(roi_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop ROI edges
    roi_canny = canny_bgr[y1:y2, x1:x2]
    canny_display = np.zeros_like(enhanced)
    canny_display[y1:y2, x1:x2] = roi_canny

    # Combine original + ROI edges
    combined = np.hstack((roi_img, canny_display))

    # Show frame
    cv2.imshow("Center ROI Video", combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
