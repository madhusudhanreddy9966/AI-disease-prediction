import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import pickle

class SkinDiseaseClassifier:
    def __init__(self, img_size=(224, 224), num_classes=7):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def build_model(self):
        """Build CNN model for skin disease classification"""
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def preprocess_data(self, data_dir):
        """Load and preprocess training data"""
        images = []
        labels = []
        
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path)[:100]:  # Limit for faster training
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
                        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(class_name)
                    except:
                        continue
        
        images = np.array(images)
        labels = self.label_encoder.fit_transform(labels)
        labels = tf.keras.utils.to_categorical(labels, self.num_classes)
        
        return images, labels
    
    def train(self, data_dir, epochs=10):
        """Train the model"""
        X, y = self.preprocess_data(data_dir)
        self.build_model()
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32
        )
        
        return history
    
    def predict(self, image):
        """Predict disease from image"""
        if isinstance(image, str):
            img = tf.keras.preprocessing.image.load_img(image, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        else:
            # Handle PIL Image or numpy array
            if hasattr(image, 'resize'):  # PIL Image
                img = image.resize(self.img_size)
                img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            else:  # numpy array
                import cv2
                img_resized = cv2.resize(np.array(image), self.img_size)
                img_array = img_resized / 255.0
            
        img_array = np.expand_dims(img_array, axis=0)
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        disease_name = self.label_encoder.inverse_transform([predicted_class])[0]
        return disease_name, confidence
    
    def save_model(self, model_path, encoder_path):
        """Save trained model and label encoder"""
        self.model.save(model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_model(self, model_path, encoder_path):
        """Load trained model and label encoder"""
        self.model = tf.keras.models.load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)