import os
import shutil
import random
from tqdm import tqdm
import yaml

import config

# Set the seed for reproducibility
random.seed(42)

# Define paths

archive_dir = config.FACES_ARCHIVE_DIR

scripts_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(scripts_dir, 'data.yaml')

source_images_dir = os.path.join(archive_dir, 'images')
source_labels_dir = os.path.join(archive_dir, 'labels')

output_dir = os.path.dirname(archive_dir)

train_images_dir = os.path.join(output_dir, 'train', 'images')
train_labels_dir = os.path.join(output_dir, 'train', 'labels')
val_images_dir = os.path.join(output_dir, 'val', 'images')
val_labels_dir = os.path.join(output_dir, 'val', 'labels')


# clean up the output directories
shutil.rmtree(train_images_dir, ignore_errors=True)
shutil.rmtree(train_labels_dir, ignore_errors=True)
shutil.rmtree(val_images_dir, ignore_errors=True)
shutil.rmtree(val_labels_dir, ignore_errors=True)

# Make sure the output directories exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

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

# Function to select a subset of the dataset
def select_subset(images, percentage):
    num_images = int(len(images) * (percentage / 100))
    return random.sample(images, num_images)

# Select subset of the dataset
train_images = select_subset(train_images, config.DATASET_PERCENTAGE)
val_images = select_subset(val_images, config.DATASET_PERCENTAGE)

# Function to move files
def move_files(file_list, source_dir, target_dir):
    for file_name in tqdm(file_list):
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# Move training files
move_files(train_images, source_images_dir, train_images_dir)
move_files([f.replace('.jpg', '.txt') for f in train_images], source_labels_dir, train_labels_dir)

# Move validation files
move_files(val_images, source_images_dir, val_images_dir)
move_files([f.replace('.jpg', '.txt') for f in val_images], source_labels_dir, val_labels_dir)

# Generate data.yaml
data_yaml = {
    'train': train_images_dir,
    'val': val_images_dir,
    'nc': 1,
    'names': ['face']
}

# Save data.yaml
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f)

print("Dataset split completed.")
