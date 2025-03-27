# 💊 Medical Cost Prediction Using Machine Learning

This project aims to analyze and predict individual medical insurance costs based on demographic and lifestyle attributes using machine learning models. The dataset includes variables like age, BMI, number of children, smoking status, and region.

---

## 📁 Dataset

👉 Download the `medical_costs.csv` dataset from Repository

---

## 📊 Project Overview

We use a structured pipeline including:

- Exploratory Data Analysis (EDA) with visualizations
- Preprocessing with encoding & feature scaling
- Model Training with:
  - Linear Regression
  - Random Forest Regressor
- Performance Metrics: MAE, MSE, R² Score

---

## 🛠️ Setup Instructions

### 1. Install Python 3.11.4 and Add to PATH

Windows:  
- Download from [python.org](https://www.python.org/)
- Check "Add Python to PATH" during installation  
- Verify:
```bash
python --version
```

macOS/Linux:
```bash
brew install python@3.11
# or
sudo apt update && sudo apt install python3.11
```

---

### 2. Clone the Repository

```bash
git clone https://github.com/YourUsername/Medical-Cost-Prediction.git
cd Medical-Cost-Prediction
```

---

### 3. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

---

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

### 5. Run the Script

```bash
python software.py
```

---

## 📈 Visualizations

- Distribution of medical costs
- Relationship between Age/BMI/Smoking and Cost
- Heatmap of feature correlation

---

## 🤖 Machine Learning Models

| Model              | MAE     | MSE     | R² Score |
|-------------------|---------|---------|----------|
| Linear Regression | ~Output | ~Output | ~Output  |
| Random Forest     | ~Output | ~Output | ~Output  |

*(Actual scores are printed after execution)*

---

## 📌 File Structure

- `software.py`: Main analysis & prediction script
- `medical_costs.csv`: Dataset file
- `requirements.txt`: Library dependencies
- `README.md`: Project documentation

---
