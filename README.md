# AI-powered-digit-recognition

## OVERVIEW:

This project is a real-time digit classification system that enables users to draw numbers in the air using hand gestures for a seamless, touch-free interaction experience. It leverages CNN models, MediaPipe for hand tracking, and OpenCV for image processing, making it an innovative approach to digit recognition.

## DATASET: 
Model is trained on MNIST dataset which contains 60,000 examples of digit images and a test set of 10,000 examples. All the images are grayscale and 28x28 pixels in size.

## TECH USED:

- Programming Language: Python

- Deep Learning: TensorFlow, Keras

- Computer Vision: OpenCV, MediaPipe

- Data Processing: NumPy, Matplotlib

## HOW MODEL WORKS?

A. Hand Tracking:

- Uses MediaPipe to detect hand landmarks

- Extracts the index fingertip position (landmark 8) to track movements

- Creates a virtual drawing canvas based on hand motion

B. Digit Image Preprocessing:

- Converts to grayscale & binary

- Extracts the digit’s contour & bounding box

- Resizes the image to 28x28 pixels (same format as MNIST)

- Applies Gaussian blur & normalization for better CNN performance

C. CNN Model for Digit Classification:

Once the digit is preprocessed, it's fed into a Convolutional Neural Network (CNN) trained on the MNIST dataset

## CNN Architecture:

Conv2D + ReLU -> Extracts spatial features

MaxPooling -> Reduces dimensions while keeping key features

Flatten + Dense Layers -> Final classification layers

Softmax Activation -> Predicts the digit from 0-9

## Controls:

To run the project, open the display.py file and run the following commands on the terminal:

-> python display.py

Drawing Controls:
- d -> Start drawing
- s -> Stop drawing
- c -> Clear drawing
- Enter -> Predict the digit









  
