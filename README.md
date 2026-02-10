# Rock–Paper–Scissors Image Classification

A deep learning–based image classification system that recognizes **Rock**, **Paper**, and **Scissors** hand gestures using a **Convolutional Neural Network (CNN)**.  
The project demonstrates an end-to-end computer vision pipeline, from data preprocessing to model evaluation, with strong performance on unseen data.

---

## Overview

This project implements a visual recognition system capable of classifying hand gesture images into three categories: Rock, Paper, and Scissors. The model is trained on labeled RGB images and evaluated using a held-out test set to ensure unbiased performance measurement.

The focus is on:
- Correct dataset handling and preprocessing  
- Designing an effective CNN architecture  
- Proper train–test splitting and validation  
- Transparent evaluation using standard classification metrics  

---

## Technical Stack

- **Language:** Python  
- **Frameworks:** TensorFlow, Keras  
- **Libraries:** NumPy, OpenCV, Matplotlib, scikit-learn  
- **Domain:** Computer Vision, Deep Learning  

---

## Dataset

- **Total Images:** 2,188  
- **Training Set:** ~80% (≈1,750 images)  
- **Test Set:** ~20% (≈438 images)  
- **Image Resolution:** 64 × 64 pixels  
- **Color Format:** RGB  

Images are resized and normalized to the range [0, 1] before training.

---

## Model Architecture

The classification model is a Convolutional Neural Network composed of:

- Convolutional layer (32 filters, 3×3) with ReLU activation  
- Max pooling layer (2×2)  
- Convolutional layer (64 filters, 3×3) with ReLU activation  
- Max pooling layer (2×2)  
- Flatten layer  
- Fully connected layer (128 neurons) with ReLU  
- Output layer (3 neurons) with Softmax  

**Total Parameters:** 1,625,539

---

## Training Configuration

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** 10  
- **Batch Size:** 32  

Validation is performed on unseen data after each epoch to monitor generalization.

---

## Results

- **Test Accuracy:** **97.95%**  
- **Test Loss:** 0.0748  
- **Error Rate:** 2.05%  

Model evaluation includes accuracy and loss metrics, confusion matrix analysis, and per-class precision, recall, and F1-scores.

---

## Key Learnings

- Practical implementation of CNNs for image classification  
- Image preprocessing and normalization techniques  
- Importance of train–test splitting for unbiased evaluation  
- Model evaluation using confusion matrices and classification reports  
- Building reproducible and well-documented machine learning projects  

---

## Future Improvements

- Data augmentation for improved robustness  
- Real-time gesture recognition using live input  
- Transfer learning with pre-trained CNN architectures  
- Deployment using TensorFlow Lite or TensorFlow.js  

---

## Author

 **Arka Das**  
 Machine Learning Enthusiast
