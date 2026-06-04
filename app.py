from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
import pandas as pd
from functools import wraps
from models import init_db, User
from ml_classifiers import train_all_classifiers, predict_from_csv, get_eda_stats

# ---------------- Flask App Setup ----------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'your-secret-key-here')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ---------------- User Loader ----------------
@login_manager.user_loader
def load_user(user_id):
    return User.get(int(user_id))

# ---------------- Role-based Access Decorators ----------------
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('You need admin privileges to access this page.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

def user_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'user':
            flash('This page is only accessible to regular users.', 'danger')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# ---------------- Routes ----------------
@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        mobile = request.form.get('mobile')
        email = request.form.get('email')
        address = request.form.get('address')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        
        if User.create(name, mobile, email, address, password, 'user'):
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.get_by_email(email)
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'danger')
            return redirect(url_for('login'))
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/eda')
@login_required
@admin_required
def eda():
    stats = get_eda_stats()
    return render_template('eda.html', stats=stats)

# ---------------- Main Disorder Classification ----------------
@app.route('/main_disorder_classification', methods=['GET', 'POST'])
@login_required
@admin_required
def main_disorder_classification():
    classifiers = [
        'Random Forest', 
        'Support Vector Machine', 
        'Logistic Regression', 
        'Decision Tree', 
        'Hybrid Graph Neural Network'
    ]
    
    if request.method == 'POST':
        selected_classifiers = request.form.getlist('classifiers')
        
        if not selected_classifiers:
            flash('Please select at least one classifier!', 'warning')
            return redirect(url_for('main_disorder_classification'))
        
        results = train_all_classifiers('main.disorder')
        filtered_results = {k: v for k, v in results.items() if k in selected_classifiers}
        
        return render_template(
            'classification_results.html',
            results=filtered_results,
            target='Main Disorder',
            target_column='main.disorder',
            endpoint='main_disorder_classification'  # Pass endpoint for Jinja2
        )
    
    return render_template(
        'classifier_selection.html',
        classifiers=classifiers,
        target='Main Disorder',
        endpoint='main_disorder_classification'
    )

# ---------------- Specific Disorder Classification ----------------
@app.route('/specific_disorder_classification', methods=['GET', 'POST'])
@login_required
@admin_required
def specific_disorder_classification():
    classifiers = [
        'Random Forest', 
        'Support Vector Machine', 
        'Logistic Regression', 
        'Decision Tree', 
        'Hybrid Graph Neural Network'
    ]
    
    if request.method == 'POST':
        selected_classifiers = request.form.getlist('classifiers')
        
        if not selected_classifiers:
            flash('Please select at least one classifier!', 'warning')
            return redirect(url_for('specific_disorder_classification'))
        
        results = train_all_classifiers('specific.disorder')
        filtered_results = {k: v for k, v in results.items() if k in selected_classifiers}
        
        return render_template(
            'classification_results.html',
            results=filtered_results,
            target='Specific Disorder',
            target_column='specific.disorder',
            endpoint='specific_disorder_classification'  # Pass endpoint
        )
    
    return render_template(
        'classifier_selection.html',
        classifiers=classifiers,
        target='Specific Disorder',
        endpoint='specific_disorder_classification'
    )

# ---------------- Performance Comparison ----------------
@app.route('/performance_comparison')
@login_required
@admin_required
def performance_comparison():
    main_results = {}
    specific_results = {}
    
    if os.path.exists('model'):
        main_results = train_all_classifiers('main.disorder')
        specific_results = train_all_classifiers('specific.disorder')
    
    return render_template(
        'performance_comparison.html',
        main_results=main_results,
        specific_results=specific_results
    )

# ---------------- User Prediction ----------------
@app.route('/prediction', methods=['GET', 'POST'])
@login_required
@user_required
def prediction():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(url_for('prediction'))
        
        file = request.files['file']
        target_column = request.form.get('target_column')
        
        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(url_for('prediction'))
        
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                predictions = predict_from_csv(filepath, target_column)
                df_original = pd.read_csv(filepath)
                
                return render_template(
                    'prediction_results.html',
                    predictions=predictions,
                    target=target_column,
                    num_samples=len(df_original)
                )
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'danger')
                return redirect(url_for('prediction'))
        else:
            flash('Please upload a CSV file!', 'danger')
            return redirect(url_for('prediction'))
    
    return render_template('prediction.html')

# ---------------- Main ----------------
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
