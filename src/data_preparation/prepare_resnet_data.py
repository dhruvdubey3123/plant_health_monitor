import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Paths
RAW_TRAIN_DIR = Path('data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')
RAW_VALID_DIR = Path('data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid')
PROCESSED_DIR = Path('data/processed/resnet')

def prepare_resnet_data(use_subset=False, images_per_class=500):
    """
    Organize data for ResNet training
    Args:
        use_subset: If True, use subset for faster testing
        images_per_class: Number of images per class if using subset
    """
    print('='*80)
    print('PREPARING RESNET DATASET')
    print('='*80)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)
    
    # Get all classes
    all_classes = sorted([d.name for d in RAW_TRAIN_DIR.iterdir() if d.is_dir()])
    print(f'\nTotal classes: {len(all_classes)}')
    
    # Class to index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
    
    # Save class mapping
    with open(PROCESSED_DIR / 'class_mapping.json', 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    total_images = 0
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    print(f'\nProcessing {"subset" if use_subset else "full dataset"}...')
    
    for class_name in all_classes:
        # Get images from train directory
        train_class_dir = RAW_TRAIN_DIR / class_name
        train_images = list(train_class_dir.glob('*.jpg')) + list(train_class_dir.glob('*.JPG'))
        
        # Get images from valid directory
        valid_class_dir = RAW_VALID_DIR / class_name
        valid_images = list(valid_class_dir.glob('*.jpg')) + list(valid_class_dir.glob('*.JPG'))
        
        # Use subset if requested
        if use_subset:
            train_images = train_images[:images_per_class]
            valid_images = valid_images[:int(images_per_class * 0.2)]
        
        # Split: 70% train, 20% val, 10% test
        if len(train_images) > 10:
            train_split, test_split = train_test_split(train_images, test_size=0.1, random_state=42)
            train_split, val_split = train_test_split(train_split, test_size=0.22, random_state=42)  # 0.22 of 0.9 ≈ 0.2 total
            
            # Add validation images to val split
            val_split.extend(valid_images)
            
            # Copy images to respective directories
            for img_list, split_name in [(train_split, 'train'), (val_split, 'val'), (test_split, 'test')]:
                dest_dir = PROCESSED_DIR / split_name / class_name
                dest_dir.mkdir(exist_ok=True)
                
                for img_path in img_list:
                    shutil.copy(img_path, dest_dir / img_path.name)
                    stats[split_name] += 1
            
            total_images += len(train_split) + len(val_split) + len(test_split)
    
    print(f'\nDataset prepared:')
    print(f'  Train: {stats["train"]} images')
    print(f'  Val: {stats["val"]} images')
    print(f'  Test: {stats["test"]} images')
    print(f'  Total: {total_images} images')
    print(f'\nClass mapping saved: {PROCESSED_DIR / "class_mapping.json"}')
    
    print('='*80)
    print('RESNET DATASET PREPARATION COMPLETE!')
    print('='*80)
    
    return PROCESSED_DIR

if __name__ == '__main__':
    # For initial testing, use subset (faster)
    # Set use_subset=False for full training
    prepare_resnet_data(use_subset=True, images_per_class=200)