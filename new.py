from collections import defaultdict
from ultralytics import YOLO
import cv2
import numpy as np

# Load your custom model
model = YOLO("best.pt")  # Load your custom model

# Load the video
cap = cv2.VideoCapture("Untitled video - Made with Clipchamp (1).mp4")
#cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize a dictionary to store track history
track_history = defaultdict(list)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict with the model
    results = model.track(frame, persist=True)

    # Get the boxes and track IDs, checking for None
    boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else []
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    for box, track_id in zip(boxes, track_ids):
        x, y, w, h = box
        track = track_history[track_id]
        track.append((float(x), float(y)))  # x, y center point
        if len(track) > 30:  # retain 30 tracks for 30 frames
            track.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

    # Display the result image in a single window
    cv2.imshow('YOLO Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
