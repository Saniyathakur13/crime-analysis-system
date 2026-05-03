# 🚔 Crime Data Analysis & Prediction System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Project Overview

The **Crime Data Analysis & Prediction System** is an interactive web application that analyzes historical crime data, identifies patterns, predicts future crime types, and provides actionable insights for law enforcement agencies. The system uses **Machine Learning** (Random Forest Classifier) to predict crime types based on location, time, and day of week.

**📋 Languages & Frameworks**
Category	Technology	Version	Purpose
Primary Language	Python	3.11+	Entire project backend
Web Framework	Streamlit	1.25+	Interactive dashboard UI
Frontend (embedded)	HTML/CSS	-	Custom styling in Streamlit

### 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Real-Time Crime Prediction** | Predict crime type based on location, hour, and day |
| **Interactive Dashboards** | Live metrics and KPIs that update with filters |
| **8+ Interactive Charts** | Crime distribution, hourly patterns, seasonal trends |
| **Risk & Severity Analysis** | Identify high-risk locations and severe crime types |
| **Police Resource Optimization** | Smart patrol allocation based on crime density |
| **Trend Forecasting** | Monthly crime trend analysis |
| **Dynamic Filtering** | Filter by date, hour, location, crime type |
| **Smart Recommendations** | Actionable insights based on filtered data |

---

## 🏗️ System Architecture
┌─────────────────────────────────────────────────────────────┐
│ Crime Analysis System │
├─────────────────────────────────────────────────────────────┤
│ Frontend: Streamlit Web Interface │
│ Backend: Python + Pandas + Scikit-learn │
│ Visualization: Plotly + Matplotlib │
│ Machine Learning: Random Forest Classifier │
└─────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│ Data Pipeline │
├─────────────────────────────────────────────────────────────┤
│ data_generation.py → crime_data.csv │
│ ↓ │
│ crime_analysis.py → processed data + ML model │
│ ↓ │
│ unified_app.py → Interactive Web Dashboard │
└─────────────────────────────────────────────────────────────┘

text

---

## 📁 Project Structure
Crime-Analysis-System/
│
├── data_generation.py # Generates synthetic crime data
├── crime_analysis.py # Processes data & trains ML model
├── unified_app.py # MAIN APPLICATION - Streamlit dashboard
│
├── crime_data.csv # Raw generated data
├── processed_crime_data.csv # Processed data with features
├── crime_model.pkl # Trained Random Forest model
├── location_risk_scores.csv # Risk scores per location
├── label_encoder_crime.pkl # Encoder for crime types
├── label_encoder_loc.pkl # Encoder for locations
│
├── plots/ # Generated visualizations
│
├── requirements.txt # Python dependencies
└── README.md # This file
