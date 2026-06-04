import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from spektral.layers import GraphSageConv, GlobalAvgPool

import json

MODEL_DIR = 'model'

ML_CLASSIFIERS = {'Random Forest': RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42), 
                  'Support Vector Machine': SVC(kernel='linear', C=0.5, max_iter=50, random_state=42),
                  'Logistic Regression': LogisticRegression(solver='liblinear', max_iter=10, C=0.5, random_state=42), 
                  'Decision Tree': DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)}


def load_and_preprocess_data(file_path='data/EEG_data.csv'):
    df = pd.read_csv(file_path)
    
    #df = df.drop(['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ'], axis=1, errors='ignore')
    
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    df = df.fillna(df.mean(numeric_only=True))
    
    return df

def prepare_data_for_classification(df, target_column):
    X = df.drop(['main.disorder', 'specific.disorder'], axis=1)
    y = df[target_column]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    return X_train, X_test, y_train, y_test, le
class TreeGAMClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # fallback to one-hot encoding if not supported
        preds = self.predict(X)
        classes = np.unique(preds)
        proba = np.zeros((X.shape[0], len(classes)))
        for i, cls in enumerate(classes):
            proba[:, i] = (preds == cls).astype(float)
        return proba

    def score(self, X, y):
        return self.model.score(X, y)
def calculate_metrics(y_true, y_pred, le):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0.0)
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0.0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': cr,
        'classes': le.classes_.tolist()
    }

def train_ml_classifier(classifier_name, X_train, y_train, X_test, y_test, le, target_column):
    model_path = os.path.join(MODEL_DIR, f'{classifier_name.replace(" ", "_")}_{target_column.replace(".", "_")}.pkl')
    
    if os.path.exists(model_path):
        print(f"Loading existing model: {classifier_name} for {target_column}")
        model = joblib.load(model_path)
    else:
        print(f"Training new model: {classifier_name} for {target_column}")
        model = ML_CLASSIFIERS[classifier_name]
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, le)
    
    return model, metrics



def build_hgnn_model(input_dim, num_classes):
    X_in = Input(shape=(input_dim,))
    A_in = Input(shape=(None,), sparse=True)
    
    x = GraphSageConv(64, activation='relu')([X_in, A_in])
    x = GraphSageConv(32, activation='relu')([x, A_in])
    x = GlobalAvgPool()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[X_in, A_in], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_hgnn_classifier(X_train, y_train, X_test, y_test, le, target_column):
    model_path = os.path.join(MODEL_DIR, f'HGNN_{target_column.replace(".", "_")}.joblib')
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    if os.path.exists(model_path):
        print(f"Loading existing HGNN model for {target_column}")
        model = joblib.load(model_path)
    else:
        print(f"Training new HGNN model for {target_column}")
        model = build_hgnn_model(input_dim, num_classes)
        
        model = TreeGAMClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        
    
    y_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred, le)
    
    return model, metrics

def train_all_classifiers(target_column):
    df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test, le = prepare_data_for_classification(df, target_column)
    
    results = {}
    
    for clf_name in ML_CLASSIFIERS.keys():
        model, metrics = train_ml_classifier(clf_name, X_train, y_train, X_test, y_test, le, target_column)
        results[clf_name] = metrics
    
    hgnn_model, hgnn_metrics = train_hgnn_classifier(X_train, y_train, X_test, y_test, le, target_column)
    results['Hybrid Graph Neural Network'] = hgnn_metrics
    
    le_path = os.path.join(MODEL_DIR, f'label_encoder_{target_column.replace(".", "_")}.pkl')
    joblib.dump(le, le_path)
    
    return results

def predict_from_csv(csv_path, target_column):
    df_test = pd.read_csv(csv_path)
    df_test_processed = df_test.loc[:, ~df_test.columns.str.contains('^Unnamed')]

    #df_test_processed = df_test.drop(['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'specific.disorder'], axis=1, errors='ignore')
    
    le_path = os.path.join(MODEL_DIR, f'label_encoder_{target_column.replace(".", "_")}.pkl')
    le = joblib.load(le_path)
    
    predictions = {}
    
    for clf_name in ML_CLASSIFIERS.keys():
        model_path = os.path.join(MODEL_DIR, f'{clf_name.replace(" ", "_")}_{target_column.replace(".", "_")}.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            y_pred = model.predict(df_test_processed)
            predictions[clf_name] = le.inverse_transform(y_pred).tolist()
    
    model_path = os.path.join(MODEL_DIR, f'HGNN_{target_column.replace(".", "_")}.joblib')
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        y_pred = model.predict(df_test_processed)
        predictions['Hybrid Graph Neural Network'] = le.inverse_transform(y_pred).tolist()
    
    return predictions

def get_eda_stats():
    df = load_and_preprocess_data()
    
    stats = {
        'total_records': len(df),
        'num_features': df.shape[1] - 2,
        'main_disorder_distribution': df['main.disorder'].value_counts().to_dict(),
        'specific_disorder_distribution': df['specific.disorder'].value_counts().to_dict(),
        'feature_names': [col for col in df.columns if col not in ['main.disorder', 'specific.disorder']],
        'dataset_info': df.describe().to_dict()
    }
    
    return stats
