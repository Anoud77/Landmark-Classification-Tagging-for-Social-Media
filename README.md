# 🏞️ Landmark Classification & Tagging for Social Media

This project applies **Convolutional Neural Networks (CNNs)** to the problem of **image-based landmark classification**, a valuable task for photo-sharing platforms that want to infer image locations from content alone. You’ll build two classifiers: one from scratch and one using transfer learning. The best model is deployed in a simple inference app.

This project is part of the **AWS Machine Learning Engineer Nanodegree** and demonstrates an end-to-end ML pipeline using PyTorch and TorchScript.

---

## 📌 Project Overview

Photo-sharing services like Instagram or Google Photos benefit from knowing the location where an image was taken. However, many images lack GPS metadata. In such cases, identifying the location from visible landmarks becomes critical.

This project tackles that problem using deep learning techniques by:
- Building a CNN from scratch to classify landmarks
- Training a second model using **transfer learning**
- Deploying the best model as a prediction app

---

## 🧰 Tools and Technologies

- Python
- PyTorch
- Torchvision
- TorchScript
- Jupyter Notebooks
- Matplotlib
- PIL / OpenCV (for preprocessing)
- Transfer Learning (ResNet, VGG, etc.)
