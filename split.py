import os
import shutil
import random

# Set the seed for reproducibility
random.seed(42)

# Define paths
source_images_dir = '/home/stan/CVS2024/datasets/archive/images'
source_labels_dir = '/home/stan/CVS2024/datasets/archive/labels'
train_images_dir = '/home/stan/CVS2024/datasets/train/images'
train_labels_dir = '/home/stan/CVS2024/datasets/train/labels'
val_images_dir = '/home/stan/CVS2024/datasets/val/images'
val_labels_dir = '/home/stan/CVS2024/datasets/val/labels'

# Get list of all images
images = [f for f in os.listdir(source_images_dir) if f.endswith('.jpg')]

# Shuffle the images
random.shuffle(images)

# Define split ratio
split_ratio = 0.8
split_index = int(len(images) * split_ratio)

# Split images into train and val
train_images = images[:split_index]
val_images = images[split_index:]

# Function to move files
def move_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# Move training files
move_files(train_images, source_images_dir, train_images_dir)
move_files([f.replace('.jpg', '.txt') for f in train_images], source_labels_dir, train_labels_dir)

# Move validation files
move_files(val_images, source_images_dir, val_images_dir)
move_files([f.replace('.jpg', '.txt') for f in val_images], source_labels_dir, val_labels_dir)

print("Dataset split completed.")