import os
import cv2
from ultralytics import YOLO
import config
# Load the YOLOv8 model

scripts_dir = os.path.dirname(os.path.abspath(__file__))
faces_checkpoint = os.path.join(scripts_dir, 'face_training', 'face', 'weights' ,'best.pt')

model_pt = faces_checkpoint if (config.DETECT_FACES and os.path.exists(faces_checkpoint)) else 'yolov8n.pt'

model = YOLO(model_pt) # hard code this to something else if you want to use a different model

def capture_video(device_index=0):
	# Initialize video capture with the device index
	cap = cv2.VideoCapture(device_index)

	if not cap.isOpened():
		print(f"Error: Could not open video device {device_index}")
		return

	print("Press 'q' to quit")
	while True:
		# Capture frame-by-frame
		ret, frame = cap.read()

		if not ret:
			print("Failed to grab frame")
			break

		# Display the resulting frame
		# cv2.imshow('Video', frame)
		results = model(frame)
		annotated_frame = results[0].plot()
		cv2.imshow('YOLOv8 Inference', annotated_frame)
		

		# Break the loop when 'q' is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture and close windows
	cap.release()
	cv2.destroyAllWindows()

# Run the video capture function
capture_video(device_index=config.CAM_DEVICE)