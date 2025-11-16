import cv2
import numpy as np

# Video path
video_path = r"D:\\Competition\\NASARoverRally\\RUET-Exover25\\camera_recording\\camera_recording.avi"

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Define middle ROI (narrowed horizontally)
ROI_Y1 = 200
ROI_Y2 = 400
ROI_X1 = 100   # Left boundary (tunable)
ROI_X2 = 400   # Right boundary (tunable)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Draw ROI rectangle
    frame_with_roi = frame.copy()
    cv2.rectangle(frame_with_roi, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2), (0, 255, 255), 2)
    cv2.putText(frame_with_roi, "ROI", (ROI_X1, ROI_Y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Create ROI mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2] = 255   # <-- only within narrowed ROI

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(masked_gray)
    
    blurred = cv2.GaussianBlur(masked_gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug_frame = frame_with_roi.copy()
    rock_data = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Distance estimation
        if area > 2000 or cy > int(ROI_Y2 - (ROI_Y2 - ROI_Y1)*0.25):
            distance = "CLOSE"
        elif area > 700 or cy > int(ROI_Y1 + (ROI_Y2 - ROI_Y1)*0.35):
            distance = "MID"
        else:
            distance = "FAR"

        # Only keep MID rocks
        if distance != "MID":
            continue

        rock_data.append({"area": area, "bbox": (x, y, w_box, h_box), "centroid": (cx, cy), "distance": distance})

        # Draw box + centroid + distance
        cv2.rectangle(debug_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.circle(debug_frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(debug_frame, distance, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Zone classification (for MID rocks only)
    one_third = (ROI_X2 - ROI_X1) // 3
    two_third = 2 * (ROI_X2 - ROI_X1) // 3
    for rock in rock_data:
        cx, cy = rock["centroid"]
        relative_cx = cx - ROI_X1
        if relative_cx < one_third:
            zone = "LEFT"
        elif relative_cx < two_third:
            zone = "CENTER"
        else:
            zone = "RIGHT"
        rock["zone"] = zone
        cv2.putText(debug_frame, zone, (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Display
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((frame_with_roi, edges_bgr, debug_frame))
    cv2.imshow("Original | Edges | MID Rocks Only", combined)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
