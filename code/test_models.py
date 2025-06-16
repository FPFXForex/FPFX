import tensorflow as tf
import numpy as np
import joblib
import os

# CONFIRMED PATH - No changes needed if files are here
MODEL_DIR = r"C:\FPFX\model"
ACTOR_PATH = os.path.join(MODEL_DIR, "actor_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def test_actor():
    print("\nTesting Actor Model...")
    try:
        # Solution for Lambda layer issue
        custom_objects = {
            'actions': tf.keras.layers.Lambda(
                lambda x: x * np.array([1.0,4.5,4.5,0.2,1.0]) + np.array([0.0,0.5,0.5,-0.1,0.0])
            )
        }
        
        # Verify file exists first
        if not os.path.exists(ACTOR_PATH):
            raise FileNotFoundError(f"File not found at: {ACTOR_PATH}")
            
        model = tf.keras.models.load_model(ACTOR_PATH, custom_objects=custom_objects)
        sample = np.random.randn(1, 1, 33).astype(np.float32)
        prediction = model.predict(sample)
        
        print(f"✅ Actor loaded! Input: {sample.shape} → Output: {prediction.shape}")
        print(f"Sample output: {prediction[0]}")
        
    except Exception as e:
        print(f"❌ Actor load failed: {str(e)}")
        print(f"Current path: {ACTOR_PATH}")
        print(f"Directory contents: {os.listdir(MODEL_DIR)}")

def test_scaler():
    print("\nTesting Scaler...")
    try:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"File not found at: {SCALER_PATH}")
            
        scaler = joblib.load(SCALER_PATH)
        sample = np.random.randn(1, 33)
        transformed = scaler.transform(sample)
        
        print(f"✅ Scaler loaded! Input: {sample.shape} → Output: {transformed.shape}")
        print(f"Sample transformation (first 5 values): {transformed[0][:5]}")
        
    except Exception as e:
        print(f"❌ Scaler load failed: {str(e)}")
        print(f"Current path: {SCALER_PATH}")

if __name__ == "__main__":
    print(f"\nCurrent working directory: {os.getcwd()}")
    print(f"Model directory contents: {os.listdir(MODEL_DIR)}")
    
    test_actor()
    test_scaler()