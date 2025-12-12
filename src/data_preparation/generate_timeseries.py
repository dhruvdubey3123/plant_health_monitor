import numpy as np
import pandas as pd
import json
from pathlib import Path

# Paths
OUTPUT_DIR = Path('data/processed/lstm')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Nutritional parameters
NUTRIENTS = ['Nitrogen', 'Phosphorus', 'Potassium', 'Calcium', 'Magnesium', 
             'Sulfur', 'Iron', 'Manganese', 'Zinc', 'pH']

# Disease to nutrient deficiency mapping
DISEASE_NUTRIENT_MAP = {
    'healthy': {'Nitrogen': 0.8, 'Phosphorus': 0.8, 'Potassium': 0.8, 'pH': 0.7},
    'early_blight': {'Nitrogen': 0.4, 'Potassium': 0.5, 'pH': 0.6},
    'late_blight': {'Nitrogen': 0.3, 'Calcium': 0.4, 'pH': 0.5},
    'leaf_spot': {'Nitrogen': 0.5, 'Magnesium': 0.4, 'Iron': 0.5},
    'rust': {'Iron': 0.3, 'Manganese': 0.4, 'Zinc': 0.5},
    'scab': {'Calcium': 0.4, 'pH': 0.5},
    'powdery_mildew': {'Potassium': 0.4, 'Sulfur': 0.3},
}

def generate_plant_timeseries(num_plants=1000, sequence_length=30):
    """
    Generate synthetic time-series data for plant health monitoring
    
    Args:
        num_plants: Number of plant sequences to generate
        sequence_length: Number of days to simulate per plant
    
    Returns:
        X: Input sequences (plant health over time)
        y: Target values (future nutritional needs)
    """
    print('='*80)
    print('GENERATING TIME-SERIES DATA')
    print('='*80)
    
    all_sequences = []
    all_targets = []
    
    for plant_id in range(num_plants):
        # Random initial plant state
        plant_type = np.random.choice(['tomato', 'potato', 'apple', 'corn', 'pepper'])
        disease_state = np.random.choice(list(DISEASE_NUTRIENT_MAP.keys()))
        
        # Initialize nutrient levels (0-1 scale)
        nutrient_levels = {n: np.random.uniform(0.5, 1.0) for n in NUTRIENTS}
        
        sequence = []
        
        for day in range(sequence_length):
            # Simulate nutrient depletion over time
            for nutrient in NUTRIENTS:
                if nutrient in DISEASE_NUTRIENT_MAP.get(disease_state, {}):
                    # Faster depletion for disease-affected nutrients
                    depletion = DISEASE_NUTRIENT_MAP[disease_state][nutrient]
                    nutrient_levels[nutrient] *= np.random.uniform(0.95, 0.98) * depletion
                else:
                    # Normal depletion
                    nutrient_levels[nutrient] *= np.random.uniform(0.97, 0.99)
                
                # Add random variation
                nutrient_levels[nutrient] += np.random.normal(0, 0.02)
                nutrient_levels[nutrient] = np.clip(nutrient_levels[nutrient], 0, 1)
            
            # Health score (average of key nutrients)
            health_score = np.mean([
                nutrient_levels['Nitrogen'],
                nutrient_levels['Phosphorus'],
                nutrient_levels['Potassium']
            ])
            
            # Record day's state
            day_data = [health_score] + [nutrient_levels[n] for n in NUTRIENTS]
            sequence.append(day_data)
        
        # Target: nutrient needs for next 7 days
        target = []
        for nutrient in NUTRIENTS:
            if nutrient_levels[nutrient] < 0.4:
                target.append(2)  # High need
            elif nutrient_levels[nutrient] < 0.6:
                target.append(1)  # Medium need
            else:
                target.append(0)  # Low/no need
        
        all_sequences.append(sequence)
        all_targets.append(target)
        
        if (plant_id + 1) % 200 == 0:
            print(f'Generated {plant_id + 1}/{num_plants} plant sequences...')
    
    X = np.array(all_sequences)
    y = np.array(all_targets)
    
    print(f'\nDataset shape:')
    print(f'  X (sequences): {X.shape}')
    print(f'  y (targets): {y.shape}')
    print(f'  Features: Health Score + {len(NUTRIENTS)} nutrients')
    
    # Save data
    np.save(OUTPUT_DIR / 'X_sequences.npy', X)
    np.save(OUTPUT_DIR / 'y_targets.npy', y)
    
    # Save metadata
    metadata = {
        'num_plants': num_plants,
        'sequence_length': sequence_length,
        'num_features': len(NUTRIENTS) + 1,
        'nutrients': NUTRIENTS,
        'target_classes': ['Low Need', 'Medium Need', 'High Need']
    }
    
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f'\nData saved to: {OUTPUT_DIR}')
    print('='*80)
    print('TIME-SERIES DATA GENERATION COMPLETE!')
    print('='*80)
    
    return X, y

if __name__ == '__main__':
    X, y = generate_plant_timeseries(num_plants=1000, sequence_length=30)