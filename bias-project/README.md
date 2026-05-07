# ML Bias Detection & Mitigation Minim-Project 🚀

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)

A complete end-to-end Machine Learning web application designed to detect and mitigate demographic bias in hiring models. Built for college mini-projects and placement showcases, this project demonstrates an understanding of **AI Ethics, Data Science, and Full-Stack Development**.

---

## 🎯 Features

- **CSV Dataset Upload:** Easily upload custom tabular hiring datasets.
- **Automated Model Training:** Instantly trains a Logistic Regression algorithm.
- **Fairness Metrics Evaluation:** Calculates industry-standard bias metrics:
  - **Statistical Parity Difference (SPD)**
  - **Disparate Impact (DI)**
- **Automated Mitigation:** Applies the *Fairness through Unawareness* technique by actively blinding the model to sensitive attributes.
- **Visual Comparisons:** Generates dynamic Matplotlib bar charts comparing selection rates before and after mitigation.
- **Modern UI/UX:** Clean, responsive, and professional frontend built using Bootstrap 5.

---

## 🧠 Core Concepts Evaluated (Viva Prep)

1. **What is Statistical Parity Difference (SPD)?**
   - The difference in selection rates between the unprivileged group (e.g., Females) and the privileged group (e.g., Males). An ideal unbiased model has an SPD of exactly `0.0`.
   
2. **What is Disparate Impact (DI)?**
   - The ratio of selection rates between the unprivileged and privileged groups. The industry and legal standard (the "80% rule") dictates that an acceptable DI falls between `0.8` and `1.25`.
   
3. **What is Fairness through Unawareness (Mitigation)?**
   - A data pre-processing technique where the sensitive attribute (like `Gender` or `Race`) is entirely removed from the dataset before training, preventing the model from overtly correlating outcomes with that specific trait.

---

## 🗂 Project Structure

```
bias-project/
│
├── app.py                  # Main Flask application and server routing
├── model.py                # Machine Learning model training logic
├── bias_detection.py       # Fairness metric calculations and Matplotlib plots
├── dataset.csv             # Sample dataset for immediate testing
├── templates/
│   ├── index.html          # Upload page & project description
│   └── results.html        # Dashboard showing metrics and graphs
├── static/
│   └── style.css           # Custom CSS styling
└── README.md               # Project documentation
```

---

## 💻 Installation & Setup

1. **Clone or Download the Repository**
2. **Navigate to the Project Directory:**
   ```bash
   cd bias-project
   ```
3. **Install Dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install flask pandas scikit-learn matplotlib
   ```
4. **Run the Application:**
   ```bash
   python3 app.py
   ```
5. **Open your Browser:**
   Visit `http://127.0.0.1:5000`

---

## 📊 Sample Dataset Format

If you wish to upload your own dataset, ensure it is a `.csv` file containing at least the following numerical columns:
- `Gender`: The sensitive attribute (`0` = Female, `1` = Male).
- `Hired`: The target target label (`0` = Not Hired, `1` = Hired).
- Other contextual features (e.g., `Experience`, `Test_Score`).

*Developed as an educational showcase for Fair AI coding practices.*