# AI Skin Disease Detector with AR Visualization

An AI-powered skin disease classification system with augmented reality visualization features.

## Features

- **7 Disease Categories Detection:**
  - Acne and Rosacea
  - Atopic Dermatitis
  - Bacterial Infections (Cellulitis, Impetigo)
  - Eczema
  - Pigmentation Disorders
  - Psoriasis and Lichen Planus
  - Seborrheic Keratoses and Benign Tumors

- **AR Visualization:**
  - AR Overlay with disease information
  - 3D confidence markers
  - Treatment recommendations
  - Real-time confidence indicators

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train the model:**
```bash
python src/train_model.py
```

3. **Run the app:**
```bash
streamlit run app/main.py
```

## Usage

1. Upload a skin image through the web interface
2. The AI model analyzes the image and predicts the disease
3. View AR-enhanced results with:
   - Disease name and confidence score
   - Visual overlays and 3D markers
   - Treatment recommendations
   - Medical disclaimers

## Model Architecture

- CNN with 3 convolutional layers
- MaxPooling and Dropout for regularization
- Dense layers for classification
- Trained on dermnet dataset

## AR Features

- **AR Overlay:** Information boxes with disease details
- **3D Markers:** Confidence visualization with circular indicators
- **Color Coding:** Different colors for each disease category
- **Interactive:** Adjustable confidence thresholds

## Disclaimer

This tool is for educational purposes only. Always consult qualified medical professionals for proper diagnosis and treatment.