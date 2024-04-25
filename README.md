# cs5830-a6-2024
# MNIST Digit Prediction API

This repository contains a FastAPI application that uses a pre-trained Keras model to predict digits from uploaded images. It is structured to demonstrate how to build, test, and deploy a machine learning model as a web service.

## Structure

- `mnist_model.h5`: The pre-trained Keras model file for MNIST digit prediction.
- `task1.py`: The FastAPI application for MNIST digit prediction (Task 1).
- `task2.py`: The FastAPI application with an improved setup (Task 2).
- `test_task1.py`: The unit tests for `task1.py`.
- `test_task2.py`: The unit tests for `task2.py`.
- `train_model.py`: Script for training the model (if applicable).
- `screenshots_report_A6.pdf`: Report containing screenshots and explanations of the application's functionality.
- `task1_screenshots`: Directory containing screenshots of task 1 outputs.
- `task2_screenshots`: Directory containing screenshots of task 2 outputs.
- `mnist_samples`: Sample MNIST images for task1.
- `hand_drawn_samples`: Hand-drawn digit images for task2.
- `test_samples`: test samples for unit tests.
- `README.md`: This README file.

# To run app (in terminal):
python task1.py mnist_model.h5
# or
python task2.py mnist_model.h5

# To run tests:
python test_task1.py
# or
python test_task2.py
