import pandas as pd
import joblib
import yaml
import os
import sys

def load_config(config_path="config/config.yaml"):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def run_inference(model_path, data_path, output_path, target_col):
    """Loads model and data, runs predictions, and saves results."""
    
    print(f"Loading production model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found. Please run src/train.py first.")
    
    pipeline = joblib.load(model_path)
    
    print(f"Loading data for inference from: {data_path}")
    df = pd.read_csv(data_path)
    
    # We keep a copy for the final output (especially customerID)
    output_df = df.copy()
    
    # Pre-processing steps similar to prepare_data in training
    # Note: The pipeline's ColumnTransformer handles the feature scaling/encoding,
    # but we must ensure column dropping and numeric conversions match.
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
    else:
        X = df
        
    # Ensure TotalCharges is numeric as expected by the pipeline's logic
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

    print("Generating predictions...")
    # Predict probabilities and classes
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)
    
    # Append to output dataframe
    output_df['Churn_Probability'] = probabilities
    output_df['Churn_Prediction'] = predictions
    
    # Reorder columns to put ID and predictions at the front for easier BI consumption
    cols = ['customerID', 'Churn', 'Churn_Prediction', 'Churn_Probability']
    other_cols = [c for c in output_df.columns if c not in cols]
    output_df = output_df[cols + other_cols]
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Inference complete. Results saved to: {output_path}")

def main():
    config = load_config()
    
    run_inference(
        model_path=config['paths']['model_output'],
        data_path=config['paths']['raw_data'],
        output_path=config['paths']['predictions_output'],
        target_col=config['data_params']['target_column']
    )

if __name__ == "__main__":
    main()
