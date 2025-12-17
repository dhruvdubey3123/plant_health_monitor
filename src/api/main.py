from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from api.inference import predict_disease, predict_nutrition

# Initialize FastAPI app
app = FastAPI(
    title="Plant Health Monitor API",
    description="AI-powered plant disease detection and nutritional prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Plant Health Monitor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - Upload image for disease detection",
            "nutrition": "/nutrition - Get nutritional recommendations",
            "health": "/health - API health check"
        },
        "models": {
            "disease_classification": "ResNet50 - 90.82% accuracy",
            "nutrition_prediction": "LSTM - 69.4% accuracy"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": True
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Image file (JPG, PNG)
    
    Returns:
        Disease prediction with confidence scores
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get prediction
        result = predict_disease(image)
        
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/nutrition")
async def nutrition():
    """
    Get nutritional recommendations
    
    Returns:
        Nutritional needs predictions for 10 nutrients
    """
    try:
        # Get nutrition predictions (uses mock data for demo)
        result = predict_nutrition()
        
        return {
            "status": "success",
            "predictions": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Complete analysis: disease detection + nutritional recommendations
    
    Args:
        file: Image file (JPG, PNG)
    
    Returns:
        Disease prediction + nutritional recommendations
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get disease prediction
        disease_result = predict_disease(image)
        
        # Get nutrition prediction
        nutrition_result = predict_nutrition()
        
        return {
            "status": "success",
            "filename": file.filename,
            "disease_analysis": disease_result,
            "nutritional_analysis": nutrition_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)