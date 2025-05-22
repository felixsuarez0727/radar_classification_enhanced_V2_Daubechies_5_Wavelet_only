# Radar Signal Classification Enhanced V2 - Daubechies 5 Wavelet Only

## Overview
This project implements a radar signal classification system using a Dense Neural Network (DNN) with Daubechies 5 wavelet features. The system is designed to classify different types of radar signals with high accuracy, achieving a 95.47% overall test accuracy.

## Features
- **Wavelet-based Feature Extraction**: Uses Daubechies 5 wavelet transform with level 3 decomposition
- **Neural Network Architecture**: Dense neural network with anti-overfitting measures
- **Advanced Training Features**:
  - Early stopping
  - Learning rate reduction
  - Class weighting for imbalanced data
  - Cross-validation support
- **Comprehensive Analysis Tools**:
  - Confusion matrix visualization
  - Training history plots
  - Class distribution analysis
  - AM signal specific analysis

## Project Structure
```
├── main.py                 # Main entry point
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # Neural network model definition
│   ├── train.py           # Training and evaluation logic
│   └── utils.py           # Visualization and utility functions
├── results/
│   ├── models/            # Saved model checkpoints
│   ├── plots/             # Generated visualizations
│   └── logs/              # Training logs and metrics
└── requirements.txt       # Project dependencies
```

## Model Performance
- **Overall Test Accuracy**: 95.47%
- **Class-wise Performance**:
  - AM_combined: 94.17% recall
  - BPSK_SATCOM: 98.94% recall
  - FMCW_Radar Altimeter: 98.60% recall
  - PULSED_Air-Ground-MTI: 79.85% recall
  - PULSED_Airborne-detection: 99.71% recall
  - PULSED_Airborne-range: 99.82% recall
  - PULSED_Ground mapping: 99.84% recall

## Signal Processing
The project uses Daubechies 5 wavelet transform for feature extraction:
- Level 3 decomposition
- Signal normalization
- Feature vector concatenation

## Neural Network Architecture
- Input layer for wavelet features
- Dense layers with ReLU activation
- Batch normalization
- Dropout for regularization
- L2 regularization
- Softmax output layer

## Training Features
- Early stopping with patience=15
- Learning rate reduction on plateau
- Class weights for imbalanced data
- Cross-validation support
- Model checkpointing

## Usage
```bash
python main.py --train_dataset path/to/train.h5 [options]

Options:
  --train_dataset TRAIN_DATASET
                        Path to training HDF5 dataset
  --test_dataset TEST_DATASET
                        Path to test HDF5 dataset
  --data_percentage DATA_PERCENTAGE
                        Percentage of data to use (0.0 to 1.0)
  --samples_per_class SAMPLES_PER_CLASS
                        Number of samples per class
  --model_type {tf,rf}  Model type: tf (TensorFlow) or rf (Random Forest)
  --epochs EPOCHS       Number of training epochs
  --batch_size BATCH_SIZE
                        Training batch size
  --cross_validation   Use cross-validation
  --cv_splits CV_SPLITS
                        Number of cross-validation splits
  --combine_am         Combine AM-related signals
  --no_class_weights   Disable class weights
  --no_early_stopping  Disable early stopping
```

## Dependencies
- Python 3.x
- TensorFlow
- NumPy
- PyWavelets
- scikit-learn
- h5py
- matplotlib
- seaborn

## Results
The model achieves high accuracy across most signal types, with particular strength in PULSED signal classification. The main challenge lies in the PULSED_Air-Ground-MTI class, which shows some confusion with AM signals.

## Future Improvements
1. Implement additional feature extraction methods:
   - FFT-based features
   - Spectrogram analysis
   - Time-domain statistics
2. Experiment with different wavelet types
3. Enhance model architecture for better PULSED_Air-Ground-MTI classification
4. Add real-time signal processing capabilities

## License
