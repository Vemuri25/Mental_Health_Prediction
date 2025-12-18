# 🧠 Mental Health Mood Prediction System

This project implements a **machine learning–based mental health mood prediction system** that estimates a user's **Mood Score** using behavioral and lifestyle data. The system includes **regression-based prediction, model comparison, ONNX deployment, and a live web interface**.

---

## 🚀 Live Deployment

The regression model is deployed as an **interactive web application** using **Gradio and ONNX Runtime**.

🔗 **Live Demo:**  
https://b677a41b58db9fcd71.gradio.live/

Users can input values and receive:
- A numerical **Mood Score**
- A qualitative **Mood Status**

---

## 📊 Project Description

This repository contains **two modeling approaches**:

### 🔹 Regression (Primary – Deployed)
- Predicts a **continuous Mood Score**
- Models:
  - PyTorch Neural Network
  - XGBoost Regressor
- Deployed using **ONNX Runtime + Gradio**

### 🔹 Classification (Secondary – Analysis Only)
- File: `classification.ipynb`
- Used for comparison with regression
- Outputs discrete mental health classes

> The deployed system is **regression-based**.  
> Classification is included only for analytical comparison.

---

## 🧾 Input Features

| Feature | Description |
|------|------------|
| Screen Time (hrs) | Daily screen usage |
| Social Media Platforms | Number of platforms used |
| Hours on TikTok | Time spent on TikTok |
| Sleep Hours | Average daily sleep |
| Stress Level | Self-reported stress score |

---

## 🧠 Model Outputs

### Mood Score
- Continuous numeric value
- Higher score → more stable or positive mood

### Mood Status Interpretation

| Mood Score Range | Status |
|-----------------|--------|
| ≥ 8.5 | 😊 Good / Stable |
| 6.0 – 8.49 | 😐 Moderate |
| < 6.0 | ⚠️ Low / At Risk |

---

## 🛠️ Technologies Used

- Python
- PyTorch
- XGBoost
- ONNX & ONNX Runtime
- Gradio
- Google Colab
- NumPy, Scikit-learn, Matplotlib

---

## 📁 Repository Structure

├── regression.ipynb
├── classification.ipynb
├── DLnet_MentalHealth.onnx
├── xgboost_MentalHealth.onnx
├── README.md


## 📦 Deployment Summary

- Models trained using PyTorch and XGBoost
- Exported to ONNX format
- Deployed in Google Colab using Gradio
- Accessible via a public web link

---

## ⚠️ Academic Disclaimer

This project is intended **for educational and analytical purposes only** and does **not** provide medical diagnosis or treatment recommendations.

---

## 👤 Author

**Vemuri Charan**  
Master’s in Applied Artificial Intelligence
