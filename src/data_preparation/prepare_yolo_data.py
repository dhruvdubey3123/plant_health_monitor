import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Paths
TRAIN_DIR = Path('data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')
YOLO_DIR = Path('data/processed/yolo')

def prepare_yolo_dataset(num_classes=5, images_per_class=100):
    """
    Prepare a subset of data in YOLO format for leaf detection
    Since images are pre-cropped, we'll use them for classification practice
    """
    print('='*80)
    print('PREPARING YOLO DATASET')
    print('='*80)
    
    # Create YOLO directory structure
    for split in ['train', 'val']:
        (YOLO_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Get subset of classes
    all_classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    selected_classes = random.sample(all_classes, min(num_classes, len(all_classes)))
    
    print(f'\nSelected {len(selected_classes)} classes:')
    for i, cls in enumerate(selected_classes):
        print(f'  {i}: {cls}')
    
    # Create class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    all_images = []
    all_labels = []
    
    # Collect images from selected classes
    for class_name in selected_classes:
        class_dir = TRAIN_DIR / class_name
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG'))
        selected_images = random.sample(images, min(images_per_class, len(images)))
        
        for img_path in selected_images:
            all_images.append(img_path)
            all_labels.append(class_to_idx[class_name])
    
    print(f'\nTotal images collected: {len(all_images)}')
    
    # Split into train/val
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f'Train images: {len(train_imgs)}')
    print(f'Val images: {len(val_imgs)}')
    
    # Copy images and create label files
    def process_split(images, labels, split):
        for img_path, label in zip(images, labels):
            # Copy image
            dest_img = YOLO_DIR / split / 'images' / img_path.name
            shutil.copy(img_path, dest_img)
            
            # Create label file (full image detection)
            # Format: class x_center y_center width height (normalized)
            label_file = YOLO_DIR / split / 'labels' / f'{img_path.stem}.txt'
            with open(label_file, 'w') as f:
                # Full image bounding box (center at 0.5, 0.5, full width/height)
                f.write(f'{label} 0.5 0.5 1.0 1.0\n')
    
    process_split(train_imgs, train_labels, 'train')
    process_split(val_imgs, val_labels, 'val')
    
    # Create dataset YAML file
    yaml_content = {
        'path': str(YOLO_DIR.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(selected_classes),
        'names': selected_classes
    }
    
    yaml_path = YOLO_DIR / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f'\nDataset YAML saved: {yaml_path}')
    print('='*80)
    print('YOLO DATASET PREPARATION COMPLETE!')
    print('='*80)
    
    return yaml_path

if __name__ == '__main__':
    # Prepare subset: 5 classes, 100 images per class
    yaml_path = prepare_yolo_dataset(num_classes=5, images_per_class=100)
    print(f'\nReady to train! Use: {yaml_path}')