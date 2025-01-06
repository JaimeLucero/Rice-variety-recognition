import json
from tensorflow.keras.models import model_from_json, load_model
import tensorflow as tf

def fix_model_config(h5_model_path, fixed_h5_model_path):
    # Load the model
    try:
        model = tf.keras.models.load_model(h5_model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Extract model configuration
    model_config = model.to_json()

    # Parse JSON configuration
    model_config_dict = json.loads(model_config)

    # Iterate through layers and remove unsupported 'groups' argument
    for layer in model_config_dict['config']['layers']:
        if layer['class_name'] == 'DepthwiseConv2D' and 'groups' in layer['config']:
            print(f"Fixing layer '{layer['config']['name']}' by removing 'groups'.")
            del layer['config']['groups']

    # Save fixed configuration to JSON
    fixed_model_config_json = json.dumps(model_config_dict)

    # Rebuild model from fixed configuration
    fixed_model = model_from_json(fixed_model_config_json)

    # Load weights from the original model
    fixed_model.load_weights(h5_model_path)

    # Save the fixed model
    fixed_model.save(fixed_h5_model_path)
    print(f"Fixed model saved at: {fixed_h5_model_path}")

# Paths
original_h5_model = 'best_model.h5'  # Path to the original .h5 model
fixed_h5_model = 'best_model_fixed.h5'  # Path to save the fixed .h5 model

# Fix the model
fix_model_config(original_h5_model, fixed_h5_model)

# Test loading the fixed model
try:
    test_model = tf.keras.models.load_model(fixed_h5_model)
    print("Fixed model loaded successfully!")
except Exception as e:
    print(f"Error loading fixed model: {e}")
