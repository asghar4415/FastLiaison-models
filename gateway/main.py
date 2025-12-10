from fastapi import FastAPI, HTTPException
import sys
from pathlib import Path
from importlib import import_module
import importlib.util

app = FastAPI(
    title="FastLiaison AI Models Gateway",
    description="Unified gateway for all AI models - Single port, multiple endpoints",
    version="1.0.0"
)

# Add models directory to Python path
models_path = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_path))

# Dictionary to track loaded models
loaded_models = {}


def load_model_app(model_dir_name: str, app_prefix: str):
    """
    Dynamically load a model's FastAPI app and mount it
    """
    try:
        model_path = models_path / model_dir_name / "main.py"
        if not model_path.exists():
            print(f"Warning: {model_dir_name}/main.py not found, skipping...")
            return None

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(
            f"{model_dir_name}.main",
            model_path
        )
        module = importlib.util.module_from_spec(spec)

        # Add model directory to path for relative imports
        sys.path.insert(0, str(models_path / model_dir_name))
        spec.loader.exec_module(module)

        # Get the FastAPI app from the module
        if hasattr(module, 'app'):
            app.mount(app_prefix, module.app)
            loaded_models[model_dir_name] = app_prefix
            print(f"✓ Loaded model: {model_dir_name} at {app_prefix}")
            return module.app
        else:
            print(f"Warning: {model_dir_name}/main.py has no 'app' variable")
            return None
    except Exception as e:
        print(f"Error loading {model_dir_name}: {str(e)}")
        return None


# Load all models automatically
model_configs = [
    ("explainable-ai-recommendations", "/xai"),
    ("predictive-career-path-model", "/career-path"),
    ("job-email-classifier", "/email-classifier"),
]


for model_dir, prefix in model_configs:
    load_model_app(model_dir, prefix)


@app.get("/")
async def root():
    """
    Gateway home - lists all available models and their endpoints
    """
    models_info = {}

    if "explainable-ai-recommendations" in loaded_models:
        models_info["xai"] = {
            "name": "Explainable AI Job Matcher",
            "prefix": "/xai",
            "endpoints": [
                {"path": "/xai/match", "method": "POST",
                    "description": "Match student to job"},
                {"path": "/xai/predict", "method": "POST",
                    "description": "Predict match score with ML model"},
                {"path": "/xai/health", "method": "GET",
                    "description": "Health check"},
                {"path": "/xai/match-by-ids", "method": "POST",
                    "description": "Match by IDs (future)"},
            ],
            "docs": "/xai/docs"
        }

    # if "predictive-career-path-model" in loaded_models:
    #     models_info["career-path"] = {
    #         "name": "Predictive Career Path Model",
    #         "prefix": "/career-path",
    #         "endpoints": [
    #             {"path": "/career-path/predict", "method": "POST", "description": "Predict career path"},
    #         ],
    #         "docs": "/career-path/docs"
    #     }

    if "ai-skill-gap-analysis" in loaded_models:
        models_info["skill-gap"] = {
            "name": "AI-Powered Skill Gap Analysis and Learning Pathways",
            "prefix": "/skill-gap",
            "endpoints": [
                {"path": "/skill-gap/analyze", "method": "POST",
                    "description": "Analyze skill gaps and generate learning pathways"},
                {"path": "/skill-gap/pathways", "method": "POST",
                    "description": "Generate learning pathway options"},
                {"path": "/skill-gap/health", "method": "GET",
                    "description": "Health check"},
            ],
            "docs": "/skill-gap/docs"
        }

    if "job-email-classifier" in loaded_models:
        models_info["email-classifier"] = {
            "name": "Job Email Classifier",
            "prefix": "/email-classifier",
            "endpoints": [
                {"path": "/email-classifier/predict", "method": "POST",
                    "description": "Classify an email as job-related or not"},
                {"path": "/email-classifier/predict/batch", "method": "POST",
                    "description": "Classify multiple emails at once"},
                {"path": "/email-classifier/health", "method": "GET",
                    "description": "Health check"},
            ],
            "docs": "/email-classifier/docs"
        }

    return {
        "message": "Welcome to FastLiaison AI Models Gateway",
        "version": "1.0.0",
        "models": models_info,
        "gateway_docs": "/docs",
        "loaded_models_count": len(loaded_models)
    }


@app.get("/health")
async def gateway_health():
    """
    Gateway health check
    """
    return {
        "status": "healthy",
        "gateway": "operational",
        "loaded_models": list(loaded_models.keys()),
        "model_endpoints": loaded_models
    }


@app.get("/models")
async def list_models():
    """
    List all loaded models and their prefixes
    """
    return {
        "loaded_models": loaded_models,
        "available_models": len(loaded_models),
        "details": "Visit / for detailed endpoint information"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
