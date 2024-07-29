
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from torchvision.transforms import functional as F
from PIL import Image
import config

class FaceDataset(Dataset):
    def __init__(self, img_folder, label_folder, transform=None):
        self.img_folder = img_folder
        self.label_folder = label_folder
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.img_files[idx])
        label_path = os.path.join(self.label_folder, self.img_files[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        with open(label_path, 'r') as file:
            boxes = []
            for line in file.readlines():
                _, x_center, y_center, width, height = map(float, line.strip().split())
                x_center *= original_width
                y_center *= original_height
                width *= original_width
                height *= original_height
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes.append([x_min, y_min, x_max, y_max])
                
                
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target



class ResizeTransform:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, target):
        original_height, original_width = image.shape[:2]
        
        # Convert the NumPy array to a PIL Image
        image = Image.fromarray(image)
        image = F.resize(image, (self.height, self.width))
        image = F.to_tensor(image)
        
        boxes = target['boxes']
        boxes[:, 0] = boxes[:, 0] * self.width / original_width
        boxes[:, 1] = boxes[:, 1] * self.height / original_height
        boxes[:, 2] = boxes[:, 2] * self.width / original_width
        boxes[:, 3] = boxes[:, 3] * self.height / original_height
        
        # Ensure all bounding boxes are valid
        valid_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            if x_max > x_min and y_max > y_min:
                valid_boxes.append([x_min, y_min, x_max, y_max])
        
        if len(valid_boxes) > 0:
            target['boxes'] = torch.tensor(valid_boxes, dtype=torch.float32)
        else:
            target['boxes'] = torch.empty((0, 4), dtype=torch.float32)
        
        return image, target

transform = ResizeTransform(height=480, width=640)
archive_dir = config.FACES_ARCHIVE_DIR

output_dir = os.path.dirname(archive_dir)


train_images_dir = os.path.join(output_dir, 'train', 'images')
train_labels_dir = os.path.join(output_dir, 'train', 'labels')
val_images_dir = os.path.join(output_dir, 'val', 'images')
val_labels_dir = os.path.join(output_dir, 'val', 'labels')



train_dataset = FaceDataset(train_images_dir, train_labels_dir, transform)
val_dataset = FaceDataset(val_images_dir, val_labels_dir, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# import matplotlib.pyplot as plt


# def visualize_ground_truth(image, boxes):
#     image = image.permute(1, 2, 0).cpu().numpy()
#     plt.figure(figsize=(10, 5))
#     plt.imshow(image)
#     for box in boxes:
#         print(box)
#         x_min, y_min, x_max, y_max = box
#         rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
#                              fill=False, color='red', linewidth=2)
#         plt.gca().add_patch(rect)
#     plt.show()
    
    
# data_iter = iter(train_loader)
# images, targets = next(data_iter)

# for i in range(len(images)):
#     visualize_ground_truth(images[i], targets[i]['boxes'])
    
    

# exit()

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=2)  # 1 class (face) + background
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)



import torch.optim as optim
import os

# Define the directory to save the models
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    lr_scheduler.step()
    
    print(f'Epoch: {epoch+1}, Loss: {losses.item()}')
    
    # Save the model at the end of each epoch
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            # Implement your evaluation metric here

print("Training Complete")
