import os

# # List video devices
# print("Listing all video devices:")
# os.system("ls -l /dev/video*")

# # Check current user's groups
# print("\nCurrent user's groups:")
# os.system("groups")

# exit()

import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with your model path if different




cap = cv2.VideoCapture(1)


print("Press 'q' to quit")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)
    # results = model(frame)

    # # Visualize results on the frame
    # annotated_frame = results[0].plot()

    # # Display the annotated frame
    # cv2.imshow('YOLOv8 Inference', annotated_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

