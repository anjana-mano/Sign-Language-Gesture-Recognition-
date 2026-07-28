# Real-Time Sign Language Gesture Recognition

## Overview
This project is a real-time Sign Language Gesture Recognition system developed using Python, OpenCV, MediaPipe, and Machine Learning. The system detects hand gestures through a webcam and predicts the corresponding sign language gesture in real time.

## Features
- Real-time hand gesture recognition
- MediaPipe hand landmark detection
- 21 hand landmark extraction
- Landmark normalization
- Random Forest-based gesture classification
- Confidence-based prediction with "Unknown" gesture handling
- Custom dataset collection

## Technologies Used
- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn

## Project Structure
```
collect_dataset.py      # Collects gesture dataset
train_model.py          # Trains the Random Forest model
predictcopylive.py      # Real-time gesture prediction
dssetorg.csv            # Dataset
requirements.txt        # Required Python packages
README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Dataset
```bash
python collect_dataset.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run Prediction
```bash
python predictcopylive.py
```

## Machine Learning Model
- Algorithm: Random Forest Classifier
- Feature Extraction: 21 Hand Landmarks (42 normalized coordinates)
- Real-time prediction using webcam
- Confidence threshold for unknown gestures

## Future Improvements
- Dynamic sign recognition
- Two-hand gesture recognition
- Deep Learning models (LSTM/CNN)
- Sentence generation from gestures

## Author
**Anjana Mano**
M.Tech Data Science
Rajiv Gandhi Institute of Technology, Kottayam
