import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_config(config_path="config/config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(df, target_col, drop_cols):
    """Drops unnecessary columns and ensures numeric types."""
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Ensure TotalCharges is numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    X = df.drop(columns=[target_col])
    y = df[target_col].map({'Yes': 1, 'No': 0}) # Binary encoding for target
    
    return X, y

def get_preprocessing_pipeline(numeric_features, categorical_features):
    """Creates a ColumnTransformer pipeline for preprocessing."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def main():
    config = load_config()
    raw_data_path = config['paths']['raw_data']
    target_col = config['data_params']['target_column']
    drop_cols = config['data_params']['drop_columns']
    
    # Load
    df = pd.read_csv(raw_data_path)
    X, y = prepare_data(df, target_col, drop_cols)
    
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Split
    split = StratifiedShuffleSplit(n_splits=1, test_size=config['data_params']['test_size'], 
                                   random_state=config['data_params']['random_state'])
    
    for train_index, test_index in split.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Build Pipeline
    preprocessor = get_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Fit and Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names for processed data
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features_transformed = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + cat_features_transformed
    
    # Save artifacts
    os.makedirs(config['paths']['processed_data'], exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['preprocessor_path']), exist_ok=True)
    
    # Save preprocessor
    joblib.dump(preprocessor, config['paths']['preprocessor_path'])
    
    # Save processed data
    pd.DataFrame(X_train_processed, columns=all_feature_names).to_csv(os.path.join(config['paths']['processed_data'], "X_train.csv"), index=False)
    pd.DataFrame(X_test_processed, columns=all_feature_names).to_csv(os.path.join(config['paths']['processed_data'], "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(config['paths']['processed_data'], "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(config['paths']['processed_data'], "y_test.csv"), index=False)
    
    print("\nPreprocessing complete. Artifacts saved in data/processed/ and models/.")

if __name__ == "__main__":
    main()
