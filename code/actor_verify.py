import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r"C:/FPFX/model/actor.h5")
sample_input = np.random.rand(1, 1, 33)  # Simulated 33-feature observation
print("Sample prediction:", model.predict(sample_input))