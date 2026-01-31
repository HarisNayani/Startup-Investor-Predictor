import pandas as pd

def assign_invest_decision(input_path='processed_data_scaled.csv', output_path='processed_data_scaled.csv'):
    df = pd.read_csv(input_path)
    if 'is_profitable' in df.columns:
        df['Invest_Decision'] = df['is_profitable'].apply(lambda x: 1 if x == 1 else 0)
        df.to_csv(output_path, index=False)
        print(f"Invest_Decision assigned based on is_profitable. Output: {output_path}")
    else:
        print("Column 'is_profitable' not found.")

if __name__ == "__main__":
    assign_invest_decision()
