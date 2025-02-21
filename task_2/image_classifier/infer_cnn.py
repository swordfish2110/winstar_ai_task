import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

class ImageCNNWrap:
    def __init__(self, model_path, target_size):
        self.model = tf.keras.models.load_model(model_path)
        self.target_size = target_size
        self.class_indices = None
    
    def load_class_indices(self, train_generator):
        """ Load class indices from a trained generator """
        self.class_indices = {v: k for k, v in train_generator.class_indices.items()}
    
    def preprocess_image(self, img_path):
        """ Load and preprocess a single image """
        img = image.load_img(img_path, target_size=self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize data
        return img_array
    
    def predict(self, img_path):
        """ Predict the class of an input image """
        img_array = self.preprocess_image(img_path)
        prediction = self.model.predict(img_array)
        class_index = np.argmax(prediction)
        class_label = self.class_indices.get(class_index, "Unknown") if self.class_indices else class_index
        return class_label, prediction[0]