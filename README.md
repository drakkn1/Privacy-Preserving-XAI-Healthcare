Privacy-Preserving Explainable AI Framework for Secure Healthcare Data Management

This repository contains a simplified implementation of the framework proposed in the research study:

“A Privacy-Preserving Explainable AI Framework for Secure Healthcare Data Management under Cyberthreats.”

The framework integrates Federated Learning, Differential Privacy, and Explainable AI techniques to enable secure analysis of healthcare data while maintaining model interpretability.

Overview

Artificial intelligence models trained on electronic health records (EHR) can significantly improve clinical decision support systems. However, such models may expose sensitive patient information through model outputs, gradients, or explanation mechanisms. This project demonstrates a privacy-preserving learning framework that mitigates these risks while maintaining predictive performance and interpretability.

The implementation demonstrates the integration of the following components:

Federated Learning (FL) for decentralized model training

Differential Privacy (DP-SGD) for protecting model updates

SHAP-based Explainable AI for model interpretability

Explanation filtering to reduce information leakage

Attack-Value Score (AVS) for identifying high-risk features

Technologies Used

Python
TensorFlow
TensorFlow Federated
SHAP
Scikit-learn
XGBoost

Repository Structure

train_model.py — main training and evaluation script
federated_training.py — implementation of the federated learning model
explainability.py — SHAP explanation generation and filtering
requirements.txt — required Python dependencies
