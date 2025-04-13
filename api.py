import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Union
import uvicorn
import numpy as np



DEFAULT_MODEL_CONFIG = {
    "input_sequence": [],
    "health_calculation": "predict_proba",
    "health_multiplier": 100,
    "requires_tensor": False,
    "framework": "sklearn",
    "health_mapping": {
        "source": "probability",
        "class_index": 0
    }
}
app = FastAPI()

ROOT_DIR = os.getcwd()
MODELS_DIR = os.path.join(ROOT_DIR, "models")
models: Dict[str, dict] = {}

def init():
    for model_name in os.listdir(MODELS_DIR):
        model_path = os.path.join(MODELS_DIR, model_name)
        print(f"---------- Loading {model_name} ----------")
        model_data = {}

        try:
            model_data["Model"] = joblib.load(os.path.join(model_path, "model.joblib"))
            print("Main Model Loaded")
        except FileNotFoundError:
            print(f"Model file missing: {model_name}/model.joblib")
            continue

        if os.path.exists(os.path.join(model_path, "preprocessingConfig.json")):
            with open(os.path.join(model_path, "preprocessingConfig.json"), 'r') as f:
                model_data["PreprocessingConfig"] = json.load(f)
                print("PreprocessingConfig Loaded...")

        if os.path.exists(os.path.join(model_path, "modelConfig.json")):
            with open(os.path.join(model_path, "modelConfig.json"), 'r') as f:
                model_data["ModelConfig"] = json.load(f)
        else:
            print("No modelConfig.json found. Using default configuration.")
            model_data["ModelConfig"] = DEFAULT_MODEL_CONFIG

        try:
            model_data["Scaler"] = joblib.load(os.path.join(model_path, "scaler.joblib"))
            print("Scaler Loaded")
        except FileNotFoundError:
            print("No Scaler Found, continuing without it...")
        try:
            diag_model_path = os.path.join(model_path, "diagnosticModel.joblib")
            diag_scaler_path = os.path.join(model_path, "diagnosticScaler.joblib")
            diag_config_path = os.path.join(model_path, "diagnosticConfig.json")

            if os.path.exists(diag_model_path) and os.path.exists(diag_scaler_path):
                model_data["diagnosticModel"] = joblib.load(diag_model_path)
                model_data["diagnosticScaler"] = joblib.load(diag_scaler_path)

                if os.path.exists(diag_config_path):
                    with open(diag_config_path, "r") as f:
                        model_data["diagnosticConfig"] = json.load(f)
                    print("Diagnostics Loaded")
        except Exception as e:
            print(f"Failed to load diagnostics for {model_name}: {e}")

        models[model_name] = model_data
def apply_preprocessing(raw_inputs: dict, config: dict, scaler=None) -> list:
    feature_names = list(raw_inputs.keys())
    processed = list(raw_inputs.values())

    onehot_config = config.get("OneHotEncoding", {})
    new_features = []
    new_columns = []

    for feature, onehot_info in onehot_config.items():
        if feature in raw_inputs:
            value = raw_inputs[feature]
            categories = onehot_info.get("categories", [])
            columns = onehot_info.get("columns", [])

            for i, cat in enumerate(categories):
                new_features.append(1 if value == cat else 0)
                new_columns.append(columns[i])

            idx = feature_names.index(feature)
            feature_names.pop(idx)
            processed.pop(idx)

    processed += new_features
    feature_names += new_columns

    expected_features = config.get("ExpectedFeatures", [])
    if expected_features:
        processed_reordered = [processed[feature_names.index(f)] for f in expected_features]
        processed = processed_reordered
        feature_names = expected_features

    processed = np.array(processed, dtype=float).reshape(1, -1)

    scale_config = config.get("StandardScaling", {})
    scale_cols = scale_config.get("columns", [])

    if scaler and scale_cols:
        scale_indices = [feature_names.index(col) for col in scale_cols]
        processed_scaled = processed.copy()
        processed_scaled[:, scale_indices] = scaler.transform(processed[:, scale_indices])
        return processed_scaled.tolist()

    return processed.tolist()

def reorder_inputs(processed_inputs, expected_sequence):
    current_features = processed_inputs[0]
    reordered_features = [current_features[expected_sequence.index(feature)] for feature in expected_sequence]
    return np.array(reordered_features).reshape(1, -1)

@app.post("/getPred/{mod}/")
async def predict(mod: str, inputs: Dict[str, Union[str, float, int]]):
    if mod not in models:
        raise HTTPException(status_code=404, detail=f"Model '{mod}' not found")

    try:
        model_data = models[mod]
        preproc_config = model_data.get("PreprocessingConfig", {})
        model_config = model_data.get("ModelConfig", DEFAULT_MODEL_CONFIG)

        processed_inputs = apply_preprocessing(inputs, preproc_config, model_data.get("Scaler"))

        if model_config.get("requires_tensor", False):
            import tensorflow as tf
            processed_inputs = tf.convert_to_tensor(processed_inputs, dtype=tf.float32)

        model = model_data["Model"]
        framework = model_config.get("framework", "sklearn")
        health_calculation = model_config.get("health_calculation", "predict_proba")
        health_mapping = model_config.get("health_mapping", {})
        health_multiplier = model_config.get("health_multiplier", 1)

        if framework == "sklearn":
            if health_calculation == "predict_proba":
                proba = model.predict_proba(processed_inputs)[0]
                class_index = health_mapping.get("class_index", 0)
                health_score = proba[class_index] * health_multiplier
                prediction = int(health_score < 50)
            elif health_calculation == "direct":
                raw_prediction = model.predict(processed_inputs)[0]
                health_score = raw_prediction * health_multiplier
                prediction = int(health_score > health_mapping.get("threshold", 50))
            else:
                raise ValueError("Invalid health calculation method in model config")
        elif framework == "tensorflow":
            if health_calculation == "predict":
                proba = model.predict(processed_inputs)[0]
                class_index = health_mapping.get("class_index", 0)
                
                health_score = proba[class_index] * health_multiplier
                prediction = int(health_score < 50)
            elif health_calculation == "direct":
                raw_prediction = model.predict(processed_inputs)[0]
                health_score = raw_prediction * health_multiplier
                prediction = int(health_score > health_mapping.get("threshold", 50))
            else:
                raise ValueError("Invalid health calculation method in model config")
        else:
            raise ValueError("Unsupported framework in model config")

        diagnostic = "No Diagnostics Available"
        if prediction == 1 and "diagnosticModel" in model_data:
            diag_input = apply_preprocessing(inputs, preproc_config, model_data.get("diagnosticScaler"))
            diag_result = model_data["diagnosticModel"].predict(diag_input)[0]
            pred_class = 0
            for i in diag_result:
                if i == 1:
                    break
                pred_class += 1
            config = model_data.get("diagnosticConfig", {})
            diagnostic = config.get(str(pred_class), "No Diagnostics Available")

        return JSONResponse(
            content={
                "Health": float(health_score),
                "Status": "working" if prediction == 0 else "stopped",
                "Diagnostic": "Working Fine" if prediction == 0 else diagnostic
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
if __name__ == "__main__":
    init()
    uvicorn.run("__main__:app", host='0.0.0.0', port=3000, reload=False)
