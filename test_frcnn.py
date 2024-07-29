import os
import torch
import torchvision.transforms as T
import cv2
from train_frcnn import get_model
import time

scripts_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(scripts_dir, 'saved_models')

def find_latest_model_file(directory):
    model_files = [f for f in os.listdir(directory) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not model_files:
        return None
    latest_model_file = max(model_files, key=lambda f: int(f.split('_')[-1].replace('.pth', '')))
    
    print(f"Latest model file: {latest_model_file}")
    return latest_model_file


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=2, pretrained=False)
model.load_state_dict(torch.load(os.path.join(scripts_dir, 'saved_models', find_latest_model_file(models_dir))))
model.to(device)
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((480, 640)),
    T.ToTensor()
])

def draw_boxes(image, boxes, scores, threshold=0.5):
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Add score text
            text = f"{score:.2f}"
            cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

cap = cv2.VideoCapture(0)
start_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        start_inference = time.time()
        outputs = model(image)
        end_inference = time.time()
        
    inference_time = end_inference - start_inference
    # fps = 1 / inference_time

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    frame = draw_boxes(frame, boxes, scores)
    
    inference_time_text = f"Inference: {inference_time*1e3:.2f} ms"
    cv2.putText(frame, inference_time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()