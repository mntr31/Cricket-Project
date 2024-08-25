# from collections import defaultdict
# from ultralytics import YOLO
# import cv2
# import numpy as np

# # Load your custom model
# model = YOLO("best.pt")  # Load your custom model

# # Load the video
# cap = cv2.VideoCapture("Y2meta.app-Kohli, Gayle take RCB upto 2nd spot with thumping win over KXIP.mp4")

# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # Initialize a dictionary to store track history
# track_history = defaultdict(list)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Predict with the model
#     results = model.track(frame, persist=True)

#     # Get the boxes and track IDs, checking for None
#     boxes = results[0].boxes.xywh.cpu() if results[0].boxes.xywh is not None else []
#     track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

#     # Visualize the results on the frame
#     annotated_frame = results[0].plot()

#     # Plot the tracks
#     for box, track_id in zip(boxes, track_ids):
#         x, y, w, h = box
#         track = track_history[track_id]
#         track.append((float(x), float(y)))  # x, y center point
#         if len(track) > 30:  # retain 30 tracks for 30 frames
#             track.pop(0)

#         # Draw the tracking lines
#         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#         cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

#     # Display the result image in a single window
#     cv2.imshow('YOLO Detection', annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO, solutions

model = YOLO("best.pt")
names = model.model.names

cap = cv2.VideoCapture("RCBvsKXIP.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

line_pts = [(0, h//2), (w, h//2)]

# Init speed-estimation obj
speed_obj = solutions.SpeedEstimator(
    reg_pts=line_pts,
    names=names,
    view_img=True,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False)

    im0 = speed_obj.estimate_speed(im0, tracks)
    video_writer.write(im0)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
cap.release()
video_writer.release()
cv2.destroyAllWindows()