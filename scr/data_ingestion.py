import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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

def ingest_data(raw_data_path):
    """Loads dataset and performs initial health checks."""
    print(f"Loading data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Dataset not found at {raw_data_path}")
    
    df = pd.read_csv(raw_data_path)
    
    # Basic info
    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    print("\n--- Data Types ---")
    print(df.dtypes)
    
    # Check for missing values
    missing = df.isnull().sum()
    print("\n--- Missing Values ---")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values detected.")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates}")
    
    return df

def analyze_target(df, target_col):
    """Analyzes the distribution of the target variable."""
    if target_col not in df.columns:
        print(f"Warning: Target column '{target_col}' not found.")
        return
    
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    print(f"\n--- Target Distribution ({target_col}) ---")
    for val, count in counts.items():
        print(f"{val}: {count} ({percentages[val]:.2f}%)")
    
    if percentages.min() < 40:
        print("Note: Class imbalance detected.")

def save_correlation_matrix(df, output_path):
    """Generates and saves a correlation matrix heatmap."""
    # Only use numeric columns for correlation
    # Note: We convert object columns that should be numeric (like TotalCharges)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    numeric_df = df.select_dtypes(include=['number'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"\nCorrelation matrix saved to: {output_path}")

def main():
    config = load_config()
    
    # Load
    df = ingest_data(config['paths']['raw_data'])
    
    # Analyze Target
    analyze_target(df, config['data_params']['target_column'])
    
    # Save Plot
    save_correlation_matrix(df, os.path.join(config['paths']['figures_path'], "correlation_matrix.png"))
    
    print("\nData Ingestion and Initial EDA complete.")

if __name__ == "__main__":
    main()
