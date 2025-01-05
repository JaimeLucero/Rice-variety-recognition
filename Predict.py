import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

class Predict:
    def __init__(self, image, model_path='best_model.h5'):
        self.image = image
        self.model = tf.keras.models.load_model(model_path)  # Load model once

    def preprocess_image(self):
        image = Image.open(self.image)
        image = image.resize((250, 250))  # Match the input size used during training
        image_array = np.array(image)
        
        # Check if the pixel values are already in the range [0, 1]
        if np.max(image_array) < 1:  # If max pixel value is > 1, it means it's likely in [0, 255] range
            image_array = image_array / 255.0  # Normalize the image

        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        return image_array

    def predict(self):
        image = self.preprocess_image()
        self.model.trainable = False  # This ensures no training layers are applied
        predictions = self.model.predict(image, verbose=1)
        print("Predictions (raw output):", predictions)  # Log predictions

        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class]

        print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")

        class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
        
        return class_labels[predicted_class], confidence
