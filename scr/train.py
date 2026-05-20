import pandas as pd
import joblib
import yaml
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

def load_config(config_path="config/config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_processed_data(processed_path):
    """Loads the pre-split and processed data."""
    X_train = pd.read_csv(os.path.join(processed_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test, config):
    """Trains models, tunes XGBoost, and evaluates performance."""
    
    # Calculate scale_pos_weight for imbalance handling
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nClass imbalance ratio (0/1): {ratio:.2f}")

    # 1. Baseline Random Forest
    print("\n--- Training Baseline Random Forest ---")
    rf = RandomForestClassifier(random_state=config['data_params']['random_state'])
    rf.fit(X_train, y_train)
    rf_roc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Baseline RF ROC-AUC: {rf_roc:.4f}")

    # 2. XGBoost with Hyperparameter Tuning
    print("\n--- Tuning XGBoost Classifier ---")
    xgb = XGBClassifier(
        random_state=config['data_params']['random_state'],
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=ratio
    )
    
    param_grid = config['model_params']['xgb']
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # 3. Comprehensive Evaluation
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Final Model Evaluation (XGBoost) ---")
    print(classification_report(y_test, y_pred))
    print(f"Final ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Save Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Optimized XGBoost")
    
    os.makedirs(config['paths']['figures_path'], exist_ok=True)
    plt.savefig(os.path.join(config['paths']['figures_path'], "confusion_matrix.png"))
    plt.close()
    
    return best_model

def main():
    config = load_config()
    
    # Load data
    X_train, X_test, y_train, y_test = load_processed_data(config['paths']['processed_data'])
    
    # Train
    best_model = train_and_evaluate(X_train, X_test, y_train, y_test, config)
    
    # 4. Save Final Pipeline (Preprocessor + Model)
    # We load the fitted preprocessor and bundle it with the trained model
    preprocessor = joblib.load(config['paths']['preprocessor_path'])
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', best_model)
    ])
    
    os.makedirs(os.path.dirname(config['paths']['model_output']), exist_ok=True)
    joblib.dump(final_pipeline, config['paths']['model_output'])
    
    print(f"\nProduction pipeline saved to: {config['paths']['model_output']}")

if __name__ == "__main__":
    main()
