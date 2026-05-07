# Technologies & Components Used

## Core Technologies

### 1. Python 🐍
- The main programming language used to build everything
- Handles all the logic and calculations behind the scenes

### 2. Flask 🌐
- A lightweight web framework (like a blueprint for building websites)
- Creates the web server that runs locally on your computer
- Handles URL routes (`/`, `/upload`)

### 3. HTML & CSS with Bootstrap 5 🎨
- **HTML**: The structure of web pages (upload form, results page)
- **CSS & Bootstrap**: Makes it look nice with colors, buttons, cards, responsive design (works on mobile too)

### 4. Pandas 📊
- A data handling library in Python
- Reads and processes your CSV files
- Organizes data into tables for analysis

### 5. Scikit-Learn 🤖
- A machine learning library
- Uses **Logistic Regression** (a simple algorithm to make predictions)
- Splits data into training and testing sets

### 6. Matplotlib 📈
- Creates charts and visualizations
- Generates the bar graph comparing selection rates before and after

### 7. LabelEncoder 🏷️
- Part of scikit-learn
- Converts text data (like "Bachelors", "Masters") into numbers that the model can understand

---

## What the Project Does (Simply)

| Component | Purpose |
|-----------|---------|
| **Upload Page** | User uploads a CSV file with hiring data |
| **CSV Processing** | Reads the file and checks if it has required columns |
| **Model Training** | Trains 2 models: one with bias, one without |
| **Bias Metrics** | Calculates fairness scores (SPD, DI) |
| **Visualization** | Creates a graph showing the comparison |
| **Results Dashboard** | Shows metrics and explanations to the user |

---

## Key Concepts Explained Simply

### Statistical Parity Difference (SPD)
How different are hiring rates between men and women? Lower is fairness.

### Disparate Impact (DI)
A legal measure - is one group hired 80% as often as another? Yes = Fair

### Fairness through Unawareness
Remove gender from the model training, so it can't be biased by it

---

## Quick Project Summary

**In one sentence for someone asking:** 
*"I built a web app in Python that detects hiring bias in AI models using machine learning, and shows how to fix it."*

---

## Project Files Breakdown

### Backend (Python)
- **app.py** - Main Flask application, handles file uploads and routes
- **model.py** - Machine learning model training logic
- **bias_detection.py** - Fairness metric calculations and chart generation

### Frontend (HTML/CSS)
- **templates/index.html** - Upload page with project explanation
- **templates/results.html** - Dashboard showing fairness metrics and graphs
- **static/style.css** - Custom styling and responsive design

### Data
- **uploads/** - Folder where user-uploaded CSV files are saved
- **static/** - Folder where generated charts are saved

---

## How Everything Works Together

```
User Uploads CSV
     ↓
Flask receives file
     ↓
Pandas reads and encodes categorical data
     ↓
Scikit-Learn trains 2 models (original + mitigated)
     ↓
Bias metrics calculated (SPD, DI)
     ↓
Matplotlib generates comparison chart
     ↓
Results displayed on dashboard (HTML/CSS)
```

---

## Dependencies Required

```
Flask - Web framework
Pandas - Data processing
Scikit-Learn - Machine Learning
Matplotlib - Data visualization
Bootstrap 5 - Frontend framework
```

Install with:
```bash
pip install flask pandas scikit-learn matplotlib
```

---

**College Mini-Project Demonstration © 2026**
