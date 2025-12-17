import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
from pathlib import Path

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
RESNET_MODEL_PATH = Path('models/resnet/best.pth')
LSTM_MODEL_PATH = Path('models/lstm/best.pth')
CLASS_MAPPING_PATH = Path('data/processed/resnet/class_mapping.json')
LSTM_METADATA_PATH = Path('data/processed/lstm/metadata.json')

# Load class mapping
with open(CLASS_MAPPING_PATH, 'r') as f:
    class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load LSTM metadata
with open(LSTM_METADATA_PATH, 'r') as f:
    lstm_metadata = json.load(f)

# Image preprocessing
resnet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# LSTM Model Definition
class NutrientLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(NutrientLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output

# Load ResNet model
def load_resnet_model():
    """Load trained ResNet model"""
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 38)
    )
    model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load LSTM model
def load_lstm_model():
    """Load trained LSTM model"""
    input_size = lstm_metadata['num_features']
    output_size = len(lstm_metadata['nutrients']) * 3
    
    model = NutrientLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        output_size=output_size,
        dropout=0.3
    )
    model.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Initialize models
resnet_model = load_resnet_model()
lstm_model = load_lstm_model()

print(f"Models loaded successfully on {device}")

def predict_disease(image: Image.Image):
    """
    Predict plant disease from image
    
    Args:
        image: PIL Image
    
    Returns:
        dict: Prediction results with disease name, confidence, and top 3 predictions
    """
    # Preprocess image
    img_tensor = resnet_transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = resnet_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(probabilities, 3)
    
    # Get results
    top3_prob = top3_prob[0].cpu().numpy()
    top3_idx = top3_idx[0].cpu().numpy()
    
    # Format results
    predictions = []
    for prob, idx in zip(top3_prob, top3_idx):
        disease_name = idx_to_class[int(idx)]
        predictions.append({
            'disease': disease_name,
            'confidence': float(prob),
            'percentage': f"{float(prob) * 100:.2f}%"
        })
    
    return {
        'top_prediction': predictions[0],
        'top_3_predictions': predictions,
        'all_classes': len(class_to_idx)
    }

def predict_nutrition(health_sequence=None):
    """
    Predict nutritional needs based on health history
    
    Args:
        health_sequence: Optional time-series data (30 days, 11 features)
                        If None, generates mock data
    
    Returns:
        dict: Nutritional recommendations
    """
    if health_sequence is None:
        # Generate mock sequence (for demo)
        health_sequence = np.random.uniform(0.4, 0.9, (1, 30, 11))
    
    # Convert to tensor
    seq_tensor = torch.FloatTensor(health_sequence).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = lstm_model(seq_tensor)
    
    # Parse predictions
    nutrients = lstm_metadata['nutrients']
    target_classes = ['Low Need', 'Medium Need', 'High Need']
    
    recommendations = []
    for i, nutrient in enumerate(nutrients):
        nutrient_logits = outputs[0, i*3:(i+1)*3]
        probs = torch.nn.functional.softmax(nutrient_logits, dim=0).cpu().numpy()
        predicted_class = int(torch.argmax(nutrient_logits).cpu())
        
        recommendations.append({
            'nutrient': nutrient,
            'need_level': target_classes[predicted_class],
            'confidence': float(probs[predicted_class]),
            'probabilities': {
                'low': float(probs[0]),
                'medium': float(probs[1]),
                'high': float(probs[2])
            }
        })
    
    return {
        'recommendations': recommendations,
        'summary': get_nutrition_summary(recommendations)
    }

def get_nutrition_summary(recommendations):
    """Generate summary of nutritional needs"""
    high_need = [r['nutrient'] for r in recommendations if r['need_level'] == 'High Need']
    medium_need = [r['nutrient'] for r in recommendations if r['need_level'] == 'Medium Need']
    
    summary = {
        'urgent_attention': high_need,
        'monitor': medium_need,
        'action_required': len(high_need) > 0 or len(medium_need) > 2
    }
    
    return summary