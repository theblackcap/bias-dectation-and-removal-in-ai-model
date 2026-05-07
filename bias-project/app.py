import os
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.preprocessing import LabelEncoder

# Import custom modules
from model import train_and_evaluate
from bias_detection import calculate_metrics, create_plot

app = Flask(__name__)
# Secret key is required for flashing error messages in Flask
app.secret_key = 'super_secret_bias_detection_key_for_college_project'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    """Renders the main upload and explanation page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handles dataset upload, validation, model training, and bias evaluation."""
    
    # --- ERROR HANDLING: Check if a file was uploaded ---
    if 'dataset' not in request.files:
        flash('No file part found in the request.', 'danger')
        return redirect(url_for('index'))
    
    file = request.files['dataset']
    
    # --- ERROR HANDLING: Check if the filename is valid ---
    if file.filename == '':
        flash('No file selected for uploading.', 'danger')
        return redirect(url_for('index'))
        
    # --- ERROR HANDLING: Check if the file is a true CSV ---
    if not file.filename.endswith('.csv'):
        flash('Invalid file format. Please upload a strictly .csv file.', 'danger')
        return redirect(url_for('index'))
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # --- ERROR HANDLING: Safe CSV Data Loading ---
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Error reading the CSV file: {str(e)}", 'danger')
        return redirect(url_for('index'))
        
    # --- ERROR HANDLING: Validate required dataset columns exist ---
    req_columns = ['Gender', 'Hired']
    missing_cols = [col for col in req_columns if col not in df.columns]
    
    if missing_cols:
        flash(f"Dataset is missing required columns: {', '.join(missing_cols)}", 'danger')
        return redirect(url_for('index'))
        
    # --- ERROR HANDLING: Ensure dataset has enough rows to perform train-test splits ---
    if len(df) < 10:
        flash("Dataset is too small! Please upload data with at least 10 rows for accurate model training.", 'danger')
        return redirect(url_for('index'))
    
    # --- DATA PREPROCESSING: Encode categorical columns ---
    # Convert text columns (like Education, Gender) to numbers so the model can process them
    categorical_columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    try:
        # --- 1. MODEL TRAINING ---
        # Delegate the actual Scikit-Learn training to the imported model module
        results_base, acc_base, results_mit, acc_mit = train_and_evaluate(df)
        
        # --- 2. BIAS METRIC EVALUATION (BEFORE MITIGATION) ---
        sr_m_base, sr_f_base, spd_base, di_base = calculate_metrics(results_base)
        
        # --- 3. BIAS METRIC EVALUATION (AFTER MITIGATION) ---
        sr_m_mit, sr_f_mit, spd_mit, di_mit = calculate_metrics(results_mit)
        
        # --- 4. DATA VISUALIZATION GENERATION ---
        # --- 4. DATA VISUALIZATION GENERATION ---
        plot_filename = 'bias_comparison.png'
        plot_path = os.path.join(app.config['STATIC_FOLDER'], plot_filename)
        
        # Passes metric outputs to external function to build a dynamic Matplotlib chart
        create_plot(sr_m_base, sr_f_base, sr_m_mit, sr_f_mit, plot_path)
        plot_url = f'/static/{plot_filename}'

        # --- 5. PREPARE DASHBOARD RESPONSE ---
        data = {
            'acc_base': round(acc_base * 100, 2),
            'spd_base': round(spd_base, 3),
            'di_base': round(di_base, 3),
            
            'acc_mit': round(acc_mit * 100, 2),
            'spd_mit': round(spd_mit, 3),
            'di_mit': round(di_mit, 3),

            'plot_url': plot_url
        }
        
        return render_template('results.html', data=data)
        
    except Exception as e:
        # Catching core mathematical or machine learning calculation errors gracefully
        flash(f"An error occurred during Machine Learning processing: {str(e)}", 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Start the Flask web server locally for project demonstration
    app.run(debug=True)