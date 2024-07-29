import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


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