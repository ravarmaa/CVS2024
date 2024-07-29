import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
import cv2


def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=2)
model.load_state_dict(torch.load('/home/rando/Workshops/saved_models/model_epoch_5.pth'))
model.to(device)
model.eval()



transform = T.Compose([
    T.ToPILImage(),
    T.Resize((480, 640)),
    T.ToTensor()
])

def draw_boxes(image, boxes, scores, threshold=0.1):
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()
    
    frame = draw_boxes(frame, boxes, scores)

    cv2.imshow('Face Detector', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()