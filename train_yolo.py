import os
import shutil
from ultralytics import YOLO

scripts_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(scripts_dir, 'data.yaml')

# Clean up the output directories
shutil.rmtree(os.path.join(scripts_dir, 'face_training'), ignore_errors=True)

model = YOLO('yolov8n.pt')  # Load a pretrained model
model.train(data=yaml_path, 
            epochs=5, imgsz=1024, 
            batch=8, 
            name='face',
            project='face_training')
