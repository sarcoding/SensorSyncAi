import os
import json
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Union
import uvicorn
import numpy as np
# import warnings
# warnings.filterwarnings("ignore")



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

        # Load model-specific configuration or use default
        if os.path.exists(os.path.join(model_path, "modelConfig.json")):
            with open(os.path.join(model_path, "modelConfig.json"), 'r') as f:
                model_data["ModelConfig"] = json.load(f)
        else:
            print("No modelConfig.json found. Using default configuration.")
            model_data["ModelConfig"] = DEFAULT_MODEL_CONFIG

        # Load scaler if available
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
    # Extract feature names and values
    feature_names = list(raw_inputs.keys())
    processed = list(raw_inputs.values())
    print("ososdfoasdf", processed)

    # Apply OneHot Encoding (if applicable)
    onehot_config = config.get("OneHotEncoding", {})
    print("DFDFDFD", onehot_config)
    new_features = []  # List to hold one-hot encoded values
    new_columns = []   # List to hold new column names (features)

    for feature, onehot_info in onehot_config.items():
        if feature in raw_inputs:
            value = raw_inputs[feature]
            categories = onehot_info.get("categories", [])
            columns = onehot_info.get("columns", [])

            # Append one-hot encoded values based on the feature's value
            for i, cat in enumerate(categories):
                new_features.append(1 if value == cat else 0)
                new_columns.append(columns[i])  # Track column names for one-hot encoded features

            # Remove the original feature from processed data
            idx = feature_names.index(feature)
            print("***", feature)
            feature_names.pop(idx)
            processed.pop(idx)

    # Add the new one-hot encoded features
    processed += new_features
    print("*********", processed)
    # Update feature names with the new columns
    feature_names += new_columns

    # Ensure the final feature set matches the model's expectations
    expected_features = config.get("ExpectedFeatures", [])
    if expected_features:
        # Reorder features to match the expected order
        processed_reordered = [processed[feature_names.index(f)] for f in expected_features]
        processed = processed_reordered
        feature_names = expected_features

    # Convert to numpy array for further processing
    processed = np.array(processed, dtype=float).reshape(1, -1)
    print("dfasdfhsfhsh",processed)

    # Apply Standard Scaling (if applicable)
    scale_config = config.get("StandardScaling", {})
    print("NNNNNNNNNNN", scale_config)
    scale_cols = scale_config.get("columns", [])

    if scaler and scale_cols:
        print("Mmmmmmmsdfa")
        scale_indices = [feature_names.index(col) for col in scale_cols]
        processed_scaled = processed.copy()
        processed_scaled[:, scale_indices] = scaler.transform(processed[:, scale_indices])
        print("/////////", processed_scaled)
        return processed_scaled.tolist()

    return processed.tolist()

def reorder_inputs(processed_inputs, expected_sequence):
    current_features = processed_inputs[0]  # Flatten the input array
    print("rrrrrr", expected_sequence)
    print("mmmmmmmmmmmmm", current_features)
    reordered_features = [current_features[expected_sequence.index(feature)] for feature in expected_sequence]
    print("yyyyyyyyyyyyyy", reordered_features)
    return np.array(reordered_features).reshape(1, -1)

@app.post("/getPred/{mod}/")
async def predict(mod: str, inputs: Dict[str, Union[str, float, int]]):
    if mod not in models:
        raise HTTPException(status_code=404, detail=f"Model '{mod}' not found")

    # try:
    model_data = models[mod]
    preproc_config = model_data.get("PreprocessingConfig", {})
    model_config = model_data.get("ModelConfig", DEFAULT_MODEL_CONFIG)

    # Validate inputs against expected sequence
    processed_inputs = apply_preprocessing(inputs, preproc_config, model_data.get("Scaler"))
    print("UUUUUUUUUUUUUU", processed_inputs)
    expected_sequence = model_config.get("input_sequence", [])
    print("IIIIIIIIIII", expected_sequence)
    # if not all(feature in inputs for feature in expected_sequence):
    #     raise HTTPException(status_code=400, detail="Missing required features in input")

    # Apply preprocessing

    # Reorder inputs based on the model's expected input sequence
    # if expected_sequence:
    #     processed_inputs = reorder_inputs(processed_inputs, expected_sequence)
    print("ppppppppppppppppppppp", processed_inputs)

    # Convert inputs to tensor if required
    if model_config.get("requires_tensor", False):
        import tensorflow as tf
        processed_inputs = tf.convert_to_tensor(processed_inputs, dtype=tf.float32)

    # Predict health score
    model = model_data["Model"]
    framework = model_config.get("framework", "sklearn")
    health_calculation = model_config.get("health_calculation", "predict_proba")
    health_mapping = model_config.get("health_mapping", {})
    health_multiplier = model_config.get("health_multiplier", 1)
    print("oooooooo", health_multiplier)

    if framework == "sklearn":
        if health_calculation == "predict_proba":
            # Get probability distribution
            proba = model.predict_proba(processed_inputs)[0]
            class_index = health_mapping.get("class_index", 0)  # Default to class 0
            print("********************", proba[class_index])
            health_score = proba[class_index] * health_multiplier
            print("*****", health_score)
            prediction = int(health_score < 50)  # Example threshold for binary classification
        elif health_calculation == "direct":
            # Get raw prediction
            raw_prediction = model.predict(processed_inputs)[0]
            health_score = raw_prediction * health_multiplier
            prediction = int(health_score > health_mapping.get("threshold", 50))
        else:
            raise ValueError("Invalid health calculation method in model config")
    elif framework == "tensorflow":
        if health_calculation == "predict":
            # Get probability distribution from TensorFlow model
            proba = model.predict(processed_inputs)[0]
            class_index = health_mapping.get("class_index", 0)  # Default to class 0
            
            health_score = proba[class_index] * health_multiplier
            prediction = int(health_score < 50)  # Example threshold for binary classification
        elif health_calculation == "direct":
            # Get raw prediction from TensorFlow model
            raw_prediction = model.predict(processed_inputs)[0]
            health_score = raw_prediction * health_multiplier
            prediction = int(health_score > health_mapping.get("threshold", 50))
        else:
            raise ValueError("Invalid health calculation method in model config")
    else:
        raise ValueError("Unsupported framework in model config")

    # Determine diagnostic information
    diagnostic = "Working Fine"
    if prediction == 1 and "diagnosticModel" in model_data:
        print("())((()))")
        diag_input = model_data["diagnosticScaler"].transform(processed_inputs)
        diag_result = model_data["diagnosticModel"].predict(diag_input)[0]
        pred_class = 0
        for i in diag_result:
            if i == 1:
                break
            pred_class += 1
        # Map diagnostic result to human-readable label
        config = model_data.get("diagnosticConfig", {})
        diagnostic = config.get(str(pred_class), "No Diagnostics Available")

    return JSONResponse(
        content={
            "Health": float(health_score),
            "Status": "working" if prediction == 0 else "stopped",
            "Diagnostic": diagnostic
        }
    )

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")
    
if __name__ == "__main__":
    init()
    uvicorn.run("__main__:app", port=3000, reload=False)
# wResult = []
# fResult = []
# wrong = 0
# with open("x_test.txt", 'r') as f:
#     X_test = json.load(f)
# with open('y_test.txt', 'r') as f:
#     y_test = json.load(f)
# for x, y in zip(X_test, y_test):
#     health, result = pred("mod1", [x])
#     if result != y:
#         wrong += 1
#         wResult.append((health, result, y))
    
#     fResult.append((health, y))
# for i in fResult:
#     print(i)

# print(len(X_test), wrong, rones)
# for i in wResult:
#     print(i)


# print(pred("mod1", [[0, 298.9,	308.4,	1632,	31.8,	17]]))




# # import json

# # with open("preprocessConfig.json", 'w') as f:
# #     json.dump({"StandardScaling": {"columns": ['step_in_cycle', 'vibration_rms', 'temperature', 'pressure', 'rotational_speed', 'current']}}, f)