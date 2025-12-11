import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
TRAIN_DIR = Path('data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train')
VALID_DIR = Path('data/raw/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid')
TEST_DIR = Path('data/raw/test/test')

def explore_dataset():
    print('='*80)
    print('PLANT DISEASE DATASET EXPLORATION')
    print('='*80)
    
    # Get class distribution
    train_classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]
    print(f'\nTotal Classes: {len(train_classes)}')
    
    # Count images per class
    class_counts = {}
    for class_dir in TRAIN_DIR.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.JPG')))
            class_counts[class_dir.name] = count
    
    # Create DataFrame
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df = df.sort_values('Count', ascending=False)
    
    print(f'\nTotal Training Images: {df["Count"].sum()}')
    print(f'Average Images per Class: {df["Count"].mean():.0f}')
    print(f'Min Images per Class: {df["Count"].min()}')
    print(f'Max Images per Class: {df["Count"].max()}')
    
    # Display top 10 classes
    print('\nTop 10 Classes by Image Count:')
    print(df.head(10).to_string(index=False))
    
    # Save class distribution plot
    os.makedirs('docs', exist_ok=True)
    plt.figure(figsize=(14, 8))
    plt.barh(df['Class'][:15], df['Count'][:15], color='steelblue')
    plt.xlabel('Number of Images')
    plt.title('Top 15 Classes by Image Count')
    plt.tight_layout()
    plt.savefig('docs/class_distribution.png', dpi=300, bbox_inches='tight')
    print('\nSaved: docs/class_distribution.png')
    
    # Analyze sample images
    print('\nAnalyzing sample images...')
    sample_class = list(TRAIN_DIR.iterdir())[0]
    sample_images = list(sample_class.glob('*.jpg'))[:5]
    
    image_sizes = []
    for img_path in sample_images:
        img = cv2.imread(str(img_path))
        if img is not None:
            image_sizes.append(img.shape[:2])
    
    if image_sizes:
        avg_height = np.mean([s[0] for s in image_sizes])
        avg_width = np.mean([s[1] for s in image_sizes])
        print(f'Average Image Size: {avg_height:.0f} x {avg_width:.0f} pixels')
    
    print('\n' + '='*80)
    print('EXPLORATION COMPLETE!')
    print('='*80)
    
    return df

if __name__ == '__main__':
    df = explore_dataset()