from fastapi import FastAPI, HTTPException
import httpx

app = FastAPI(title="AI Models Gateway")

# Configure model endpoints
MODEL_ENDPOINTS = {
    "predictive-career-path-model": "http://localhost:8001/predict",
    "model2": "http://localhost:8002/predict"
}

SWAGGER_ENDPOINTS = {
    "predictive-career-path-model": "http://localhost:8001/docs",
    "model2": "http://localhost:8002/docs"
}


@app.post("/predict/{model_name}")
async def predict(model_name: str, payload: dict):
    if model_name not in MODEL_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Model not found")
    async with httpx.AsyncClient() as client:
        response = await client.post(MODEL_ENDPOINTS[model_name], json=payload)
        return response.json()


@app.get("/")
async def root():
    return {"message": "Welcome to the AI Models Gateway"}


@app.get("/models")
async def list_models():
    """
    Returns a dictionary of all models and their Swagger URLs
    """
    return SWAGGER_ENDPOINTS
