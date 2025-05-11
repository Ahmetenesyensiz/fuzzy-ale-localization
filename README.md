# Fuzzy Node Localization in Wireless Sensor Networks using Mamdani Fuzzy Inference System

## 📋 Project Description

This project aims to estimate the unknown node positions in Wireless Sensor Networks (WSNs) using a Mamdani Fuzzy Inference System (FIS).  
The approach leverages fuzzy logic principles to predict Average Localization Error (ALE) based on network parameters.  
The project was developed as part of the **BTU Computer Engineering - Soft Computing Course (2024-2025)**.

---

## 🎯 Dataset Information

Dataset Source: [UCI Machine Learning Repository - ALE in WSNs](https://archive.ics.uci.edu/dataset/844/average+localization+error+(ale)+in+sensor+node+localization+process+in+wsns)

**Features:**
- `anchor_ratio`: Ratio of anchor nodes (%)
- `trans_range`: Transmission range of sensors
- `node_density`: Number of sensor nodes
- `iterations`: Number of iterations
- `ale`: Average Localization Error (target)
- `sd_ale`: Standard deviation (not used in this project)

Total samples: **107**

---

## 🛠️ Methodology

### ✅ Membership Functions
- **Triangular**
- **Gaussian**

### ✅ Defuzzification Methods
- **Center of Sums (COS)**
- **Weighted Average (WA)**

### ✅ Total Model Combinations
- Triangular + COS
- Triangular + WA
- Gaussian + COS
- Gaussian + WA

### ✅ Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

---

## 💻 Project Structure

```text
fuzzy_node_localization_project/
├── data/                     → Dataset
├── fuzzy_models/             → Membership functions, inference engine, defuzzification
├── evaluation/               → Error metrics, visualization
├── utils/                    → Data preprocessing, rule generator
├── results/                  → Output metrics & plots (auto-generated)
├── report/                   → Project report (to be prepared by student)
├── presentation/             → YouTube link file
├── notebooks/                → (Optional) Jupyter notebooks
├── main.py                   → Main project runner
├── run_experiments.py        → (Optional) Experiment runner
├── requirements.txt          → Python dependencies
└── README.md                 → This file


