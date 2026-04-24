# 💧 Smart Water Leakage Detection System - Edge ML Simulator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Random_Forest-orange.svg)

## 📌 Project Overview
This repository contains the interactive simulation dashboard and machine learning training pipeline for a **Smart Water Leakage Detection System**. 

Instead of relying solely on reactive, threshold-based alerts, this system utilizes a **Random Forest Classifier** to analyze multiple sensor inputs simultaneously. By cross-referencing internal pipe metrics (Flow Rate, Pressure) with external environmental factors (Soil Moisture), the AI accurately differentiates between normal usage fluctuations and actual pipeline anomalies.

## ✨ Features
* **Synthetic Data Generation & Training Pipeline:** Easily generate thousands of normal and anomalous data points to train the Scikit-Learn model locally.
* **Interactive Dashboard:** Built with Streamlit, the UI allows users to manipulate sensor values via sliders to test the model's response in real-time.
* **Live ML Inference:** The dashboard feeds live slider data into the pre-trained `.pkl` model to calculate real-time leak probabilities.
* **Dynamic Visualizations:** Features custom CSS/HTML animations that visually simulate a bursting pipe when a critical leak probability (>80%) is reached.

## 🛠️ Architecture & Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn (`RandomForestClassifier`), Pandas, NumPy, Joblib
* **Frontend UI:** Streamlit, Streamlit Components (for isolated HTML/CSS animations)
* **Target Hardware (Future Scope):** Raspberry Pi (Edge Node), YF-S201 Flow Sensor, Pressure Transducer, Soil Moisture Sensor.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/smart-water-leak-sim.git](https://github.com/yourusername/smart-water-leak-sim.git)
   cd smart-water-leak-sim

1.	Create and activate a virtual environment (Recommended):

	2.	Install the required dependencies:

🎮 Usage Guide
Step 1: Train the Machine Learning Model
Before running the dashboard, you must train the model and generate the water_leak_model.pkl file.

python train_model.py

This script generates 5,000 synthetic data points, trains a Random Forest Classifier, outputs accuracy metrics to the terminal, and saves the trained model.

Step 2: Launch the Simulator
Once the .pkl file is generated, launch the interactive dashboard:

streamlit run app.py

This will open a new browser tab at http://localhost:8501. Adjust the sliders to drop the flow/pressure and spike the moisture to watch the AI trigger the leak animation.

👨‍💻 Author
Rudransh Choubey B.Tech Artificial Intelligence and Machine Learning, Semester 4
Jain University
Developed as a Literature Review & Prototype phase for the 4th-semester Smart Water Leakage Detection System project.
