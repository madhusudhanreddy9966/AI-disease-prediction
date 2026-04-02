import os
import sys

from model import SkinDiseaseClassifier

def train_model():
    """Train the skin disease classification model"""
    print("Initializing skin disease classifier...")
    classifier = SkinDiseaseClassifier()
    
    # Train the model
    train_dir = os.path.join(os.path.dirname(__file__), "..", "dataset", "train")
    train_dir = os.path.abspath(train_dir)
    
    if not os.path.exists(train_dir):
        print(f"Error: Training directory '{train_dir}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(__file__)}")
        return
    
    print("Starting training...")
    history = classifier.train(train_dir, epochs=10)
    
    # Save the model
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    models_dir = os.path.abspath(models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "skin_disease_model.h5")
    encoder_path = os.path.join(models_dir, "label_encoder.pkl")
    
    classifier.save_model(model_path, encoder_path)
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    # Print training results
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\nTraining completed!")
    print(f"Final training accuracy: {final_accuracy:.4f}")
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")

if __name__ == "__main__":
    train_model()