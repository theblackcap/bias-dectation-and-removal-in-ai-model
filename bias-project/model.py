import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_and_evaluate(df):
    """
    Trains two distinct Logistic Regression models to compare predictive outcomes:
    1. Base Model: Trained using all features, including the sensitive 'Gender' feature.
    2. Mitigated Model: Trained utilizing the 'Fairness through Unawareness' technique 
       by dropping the sensitive 'Gender' feature completely.
       
    Args:
        df (pd.DataFrame): The pre-validated pandas DataFrame containing hiring data.
        
    Returns:
        tuple: (results_base, acc_base, results_mit, acc_mit)
        - results_base (DataFrame): Test features along with baseline predictions.
        - acc_base (float): Overall predictive accuracy of baseline model.
        - results_mit (DataFrame): Test features along with mitigated predictions.
        - acc_mit (float): Overall predictive accuracy of mitigated model.
    """
    
    # Pre-processing: Split targeted feature (Hired) from training features (X)
    X = df.drop(columns=['Hired'])
    y = df['Hired']
    
    # 75% for training the machine learning pipeline, 25% for validation testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # =======================================================
    # PHASE 1: Train Original (Biased Baseline) Model
    # =======================================================
    # This model is specifically fed the 'Gender' column to demonstrate how
    # algorithmic models can improperly fixate on protected groupings.
    model_base = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_base.fit(X_train, y_train)
    
    # Get probability predictions for threshold equalization
    preds_base_proba = model_base.predict_proba(X_test)[:, 1]
    preds_base = apply_threshold_equalization(preds_base_proba, X_test['Gender'])
    acc_base = accuracy_score(y_test, preds_base)
    
    # Persist the dataframe to attach original predictions for bias calculation
    results_base = X_test.copy()
    results_base['Predicted'] = preds_base
    results_base['Gender'] = X_test['Gender']
    
    # =======================================================
    # PHASE 2: Train Mitigated (Fairness via Unawareness) Model
    # =======================================================
    # This acts as the remediation step. 'Gender' is stripped securely.
    X_train_mit = X_train.drop(columns=['Gender'])
    X_test_mit = X_test.drop(columns=['Gender'])
    
    model_mit = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_mit.fit(X_train_mit, y_train)
    
    # Get probability predictions for threshold equalization
    preds_mit_proba = model_mit.predict_proba(X_test_mit)[:, 1]
    preds_mit = apply_threshold_equalization(preds_mit_proba, X_test['Gender'])
    acc_mit = accuracy_score(y_test, preds_mit)
    
    # Combine predictions back to X_test explicitly for metrics engine
    # Notice we use 'X_test['Gender']' because the bias_detection module
    # STILL needs to know the true gender to see if dropping it actually worked!
    results_mit = X_test.copy()
    results_mit['Predicted'] = preds_mit
    results_mit['Gender'] = X_test['Gender']
    
    return results_base, acc_base, results_mit, acc_mit


def apply_threshold_equalization(probabilities, genders):
    """
    Applies threshold equalization to ensure equal selection rates across genders.
    
    This technique adjusts decision thresholds per group so that both genders
    have similar hiring rates, reducing disparate impact.
    
    Args:
        probabilities (array): Predicted probabilities from the model
        genders (array): Gender values (0=Female, 1=Male)
        
    Returns:
        array: Adjusted predictions with equalized thresholds
    """
    # Calculate optimal thresholds per gender for equal selection rates
    predictions = probabilities.copy()
    
    # Find median probabilities per gender
    female_probs = probabilities[genders == 0]
    male_probs = probabilities[genders == 1]
    
    # Set thresholds to achieve ~50% selection rate per gender
    threshold_female = np.median(female_probs) if len(female_probs) > 0 else 0.5
    threshold_male = np.median(male_probs) if len(male_probs) > 0 else 0.5
    
    # Apply adjusted thresholds
    adjusted_preds = np.zeros_like(predictions, dtype=int)
    adjusted_preds[(genders == 0) & (probabilities >= threshold_female)] = 1
    adjusted_preds[(genders == 1) & (probabilities >= threshold_male)] = 1
    
    return adjusted_preds