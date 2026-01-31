import pandas as pd


try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError("imblearn is not installed. Please install it using 'pip install imbalanced-learn'.")


def balance_dataset(input_path='processed_data_scaled.csv', output_path='processed_data_balanced.csv', target_col='Sector'):
    """Balance the dataset by Invest_Decision using SMOTE and save to output_path."""
    target_col = 'Invest_Decision'
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")
    # Drop all rows with any NaN values
    df = df.dropna()
    # Only use numeric columns for X
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in [target_col, 'Startup_Name', 'Stage']:
        if col in feature_cols:
            feature_cols.remove(col)
    X = df[feature_cols]
    y = df[target_col]
    # Set k_neighbors to min class size - 1
    class_counts = y.value_counts()
    min_class_size = class_counts.min()
    k_neighbors = max(1, min_class_size - 1)
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    balanced_df = pd.DataFrame(X_res, columns=feature_cols)
    # Add back categorical columns
    for col in ['Startup_Name', 'Sector', 'Stage']:
        if col in df.columns:
            balanced_df[col] = df[col].iloc[0]
    balanced_df[target_col] = y_res
    balanced_df.to_csv(output_path, index=False)
    print(f'Balancing complete. Output: {output_path}')

if __name__ == "__main__":
    balance_dataset()
