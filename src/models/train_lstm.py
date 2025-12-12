import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = Path('data/processed/lstm')
MODEL_DIR = Path('models/lstm')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3

class NutrientLSTM(nn.Module):
    """LSTM model for predicting nutritional needs"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(NutrientLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last time step output
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc(last_output)
        
        return output

def load_data():
    """Load and prepare time-series data"""
    print('Loading data...')
    
    X = np.load(DATA_DIR / 'X_sequences.npy')
    y = np.load(DATA_DIR / 'y_targets.npy')
    
    with open(DATA_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f'  X shape: {X.shape}')
    print(f'  y shape: {y.shape}')
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f'  Train: {X_train.shape[0]} samples')
    print(f'  Val: {X_val.shape[0]} samples')
    print(f'  Test: {X_test.shape[0]} samples')
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, metadata

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the LSTM model"""
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Multi-label classification loss
            loss = 0
            for i in range(targets.shape[1]):
                loss += criterion(outputs[:, i*3:(i+1)*3], targets[:, i])
            loss /= targets.shape[1]
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            for i in range(targets.shape[1]):
                preds = torch.argmax(outputs[:, i*3:(i+1)*3], dim=1)
                train_correct += (preds == targets[:, i]).sum().item()
                train_total += targets.shape[0]
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = 0
                for i in range(targets.shape[1]):
                    loss += criterion(outputs[:, i*3:(i+1)*3], targets[:, i])
                loss /= targets.shape[1]
                
                val_loss += loss.item()
                
                for i in range(targets.shape[1]):
                    preds = torch.argmax(outputs[:, i*3:(i+1)*3], dim=1)
                    val_correct += (preds == targets[:, i]).sum().item()
                    val_total += targets.shape[0]
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            print(f'  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / 'best.pth')
            if (epoch + 1) % 5 == 0:
                print(f'  → Best model saved! Val Acc: {val_acc:.4f}')
    
    return history, best_val_acc

def main():
    print('='*80)
    print('TRAINING LSTM FOR NUTRITIONAL NEEDS PREDICTION')
    print('='*80)
    
    print(f'\nConfiguration:')
    print(f'  Device: {device}')
    print(f'  Batch size: {BATCH_SIZE}')
    print(f'  Epochs: {NUM_EPOCHS}')
    print(f'  Learning rate: {LEARNING_RATE}')
    print(f'  Hidden size: {HIDDEN_SIZE}')
    print(f'  Num layers: {NUM_LAYERS}')
    
    # Load data
    train_loader, val_loader, test_loader, metadata = load_data()
    
    # Create model
    input_size = metadata['num_features']
    output_size = len(metadata['nutrients']) * 3  # 3 classes per nutrient
    
    print(f'\nModel architecture:')
    print(f'  Input size: {input_size}')
    print(f'  Output size: {output_size}')
    
    model = NutrientLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size,
        dropout=DROPOUT
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    print('\nStarting training...\n')
    history, best_val_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    print(f'\n{"="*80}')
    print(f'Training complete!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    print(f'{"="*80}')
    
    # Save history
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final model
    torch.save(model.state_dict(), MODEL_DIR / 'final.pth')
    
    print(f'\nModels saved:')
    print(f'  Best: {MODEL_DIR / "best.pth"}')
    print(f'  Final: {MODEL_DIR / "final.pth"}')

if __name__ == '__main__':
    main()