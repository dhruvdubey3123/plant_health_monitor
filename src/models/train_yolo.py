import sys
import torch
from pathlib import Path

# Add YOLOv5 to path
YOLO_PATH = Path('models/yolo/yolov5')
sys.path.insert(0, str(YOLO_PATH))

def train_yolo():
    """Train YOLOv5 on plant disease dataset"""
    print('='*80)
    print('TRAINING YOLOv5 MODEL')
    print('='*80)
    
    # Training parameters
    data_yaml = 'data/processed/yolo/dataset.yaml'
    img_size = 256
    batch_size = 16
    epochs = 20
    weights = 'yolov5s.pt'  # Start with small pretrained model
    
    print(f'\nConfiguration:')
    print(f'  Data: {data_yaml}')
    print(f'  Image size: {img_size}')
    print(f'  Batch size: {batch_size}')
    print(f'  Epochs: {epochs}')
    print(f'  Weights: {weights}')
    print(f'  Device: {"CUDA" if torch.cuda.is_available() else "CPU"}')
    
    # Import train module
    from train import run
    
    # Train model
    print('\nStarting training...\n')
    run(
        data=data_yaml,
        imgsz=img_size,
        batch=batch_size,
        epochs=epochs,
        weights=weights,
        project='models/yolo',
        name='plant_disease',
        exist_ok=True,
        cache=True
    )
    
    print('\n' + '='*80)
    print('TRAINING COMPLETE!')
    print('='*80)
    print('\nModel saved to: models/yolo/plant_disease/')

if __name__ == '__main__':
    train_yolo()