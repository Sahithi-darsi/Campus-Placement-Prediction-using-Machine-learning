import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    """Create necessary directories for saving results and models"""
    directories = ['results', 'models']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def load_and_prepare_data(filepath):
    """Load and prepare the dataset"""
    df = pd.read_csv(filepath)
    
    # Encode categorical variables
    gender_map = {'Male': 1, 'Female': 0}
    stream_map = {
        'Electronics And Communication': 1,
        'Computer Science': 2,
        'Information Technology': 3,
        'Mechanical': 4,
        'Electrical': 5,
        'Civil': 6,
        'Artificial Intelligence': 7
    }
    
    df['Gender'] = df['Gender'].map(gender_map)
    df['Stream'] = df['Stream'].map(stream_map)
    
    return df

def split_data(df):
    """Split the data into features and target"""
    X = df.drop('PlacedOrNot', axis=1)
    y = df['PlacedOrNot']
    return train_test_split(X, y, test_size=0.33, random_state=42)

def train_models(X_train, y_train):
    """Train different models"""
    models = {
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'GB': GradientBoostingClassifier(),
        'LGBM': LGBMClassifier(verbose=-1),  # Changed from XGBoost to LightGBM
        'CB': CatBoostClassifier(verbose=0),
        'HGB': HistGradientBoostingClassifier()
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save model
        with open(f'models/{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    return trained_models

def evaluate_model(model, X_test, y_test):
    """Evaluate a single model"""
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred) * 100,
        'precision': precision_score(y_test, y_pred) * 100,
        'recall': recall_score(y_test, y_pred) * 100,
        'f1': f1_score(y_test, y_pred) * 100,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

def save_results(model_name, metrics, results_dir='results'):
    """Save evaluation metrics to file"""
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [metrics['accuracy'], metrics['precision'], 
                 metrics['recall'], metrics['f1']]
    })
    results.to_csv(f'{results_dir}/{model_name}_results.csv', index=False)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                xticklabels=["Negative", "Positive"],
                yticklabels=['Negative', "Positive"])
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'{results_dir}/{model_name}_confusion_matrix.png')
    plt.close()

def plot_comparison(results_df):
    """Plot comparison of different metrics across models"""
    metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE']
    
    plt.figure(figsize=(15, 10))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        sns.barplot(x='Models', y=metric, data=results_df)
        plt.title(f'{metric} Comparison')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/metrics_comparison.png')
    plt.close()

def main():
    # Create necessary directories
    create_directories()
    
    # Load and prepare data
    df = load_and_prepare_data('dataset/collegePlace.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate and save results
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        save_results(name, metrics)
        
        results.append({
            'Models': name,
            'ACCURACY': metrics['accuracy'],
            'PRECISION': metrics['precision'],
            'RECALL': metrics['recall'],
            'F1_SCORE': metrics['f1']
        })
    
    # Create comparison DataFrame and plot
    final_data = pd.DataFrame(results)
    final_data.to_csv('results/final_comparison.csv', index=False)
    plot_comparison(final_data)
    
    # Print best model for each metric
    metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'F1_SCORE']
    print("\nBest Models:")
    for metric in metrics:
        best = final_data[final_data[metric] == final_data[metric].max()]
        print(f"Best {metric}: {best['Models'].values[0]} ({best[metric].values[0]:.2f}%)")

if __name__ == "__main__":
    main()