import os
import numpy as np
from gym.spaces import Box
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Lambda

# ===== UPDATED CONFIGURATION =====
MODEL_DIR = r"C:/FPFX/model"
WEIGHTS_PREFIX = os.path.join(MODEL_DIR, "ddpg_weights_actor.h5f")
OUTPUT_ACTOR = os.path.join(MODEL_DIR, "actor.h5")

# ===== ADJUSTED ARCHITECTURE =====
def build_actor(input_shape, action_space):
    state_input = Input(shape=input_shape, name="state_input")
    x = Flatten()(state_input)
    x = Dense(256, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    raw_actions = Dense(action_space.shape[0], activation="sigmoid", name="raw_actions")(x)
    diff = action_space.high - action_space.low
    low = action_space.low
    actions = Lambda(lambda x, d=diff, l=low: x * d + l, name="actions")(raw_actions)
    return Model(inputs=state_input, outputs=actions)

# ===== UPDATED SHAPE TO MATCH WEIGHTS =====
INPUT_SHAPE = (1, 33)  # Changed from 20 to 33 to match your weights
ACTION_SPACE = Box(
    low=np.array([0.0, 1.0, 1.0, -0.1, 0.0], dtype=np.float32),
    high=np.array([1.0, 10.0, 10.0, 0.1, 1.0], dtype=np.float32),
    dtype=np.float32
)

def main():
    print("[1/3] Rebuilding architecture with 33 input features...")
    actor = build_actor(INPUT_SHAPE, ACTION_SPACE)
    
    print(f"[2/3] Loading weights from:\n  {WEIGHTS_PREFIX}")
    actor.load_weights(WEIGHTS_PREFIX)
    
    print(f"[3/3] Saving to:\n  {OUTPUT_ACTOR}")
    actor.save(OUTPUT_ACTOR)
    
    print("\n✅ Success! actor.h5 rebuilt with:")
    print(f" - Input shape: {INPUT_SHAPE}")
    print(f" - Weights from: {WEIGHTS_PREFIX}")
    print("You can now use this with your trading system.")

if __name__ == "__main__":
    main()