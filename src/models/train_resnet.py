import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from pathlib import Path
import time
import copy
import json

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = Path('data/processed/resnet')
MODEL_DIR = Path('models/resnet')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
NUM_CLASSES = 38

def create_dataloaders():
    """Create train and validation dataloaders"""
    
    # Data augmentation for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create datasets
    image_datasets = {
        'train': datasets.ImageFolder(DATA_DIR / 'train', data_transforms['train']),
        'val': datasets.ImageFolder(DATA_DIR / 'val', data_transforms['val'])
    }
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, 
                          shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, 
                        shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes

def create_model():
    """Create ResNet50 model with pretrained weights"""
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )
    
    return model.to(device)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs):
    """Train the model"""
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize():5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(float(epoch_acc))
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_DIR / 'best.pth')
                print(f'  → Best model saved! Accuracy: {best_acc:.4f}')
    
    time_elapsed = time.time() - since
    print(f'\n{"="*60}')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val accuracy: {best_acc:.4f}')
    print(f'{"="*60}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save training history
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

def main():
    print('='*80)
    print('TRAINING RESNET50 FOR PLANT DISEASE CLASSIFICATION')
    print('='*80)
    
    print(f'\nConfiguration:')
    print(f'  Device: {device}')
    print(f'  Batch size: {BATCH_SIZE}')
    print(f'  Epochs: {NUM_EPOCHS}')
    print(f'  Learning rate: {LEARNING_RATE}')
    print(f'  Classes: {NUM_CLASSES}')
    
    # Create dataloaders
    print('\nLoading data...')
    dataloaders, dataset_sizes = create_dataloaders()
    print(f'  Train size: {dataset_sizes["train"]}')
    print(f'  Val size: {dataset_sizes["val"]}')
    
    # Create model
    print('\nCreating model...')
    model = create_model()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Train
    print('\nStarting training...')
    model, history = train_model(model, dataloaders, dataset_sizes, 
                                 criterion, optimizer, NUM_EPOCHS)
    
    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / 'final.pth')
    print(f'\nFinal model saved to: {MODEL_DIR / "final.pth"}')
    print(f'Best model saved to: {MODEL_DIR / "best.pth"}')

if __name__ == '__main__':
    main()