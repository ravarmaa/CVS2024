
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

import config
from frcnn_dataloader import FaceDataset, ResizeTransform
from frcnn_model import get_model


def train(epochs=5, batch_size=8):
    transform = ResizeTransform(height=480, width=640)
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    archive_dir = config.FACES_ARCHIVE_DIR
    output_dir = os.path.dirname(archive_dir)

    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')


    train_dataset = FaceDataset(train_images_dir, train_labels_dir, transform)
    val_dataset = FaceDataset(val_images_dir, val_labels_dir, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=2)  # 1 class (face) + background
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    # Define the directory to save the models
    save_dir = os.path.join(scripts_dir, 'saved_models')
    os.makedirs(save_dir, exist_ok=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        model.train()
        for images, targets in tqdm(train_loader):
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


if __name__ == '__main__':
    train()