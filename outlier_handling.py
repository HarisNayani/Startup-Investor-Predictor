
import pandas as pd
from sklearn.preprocessing import RobustScaler

def handle_outliers(input_path='processed_data.csv', output_path='processed_data_scaled.csv'):
    """Apply robust scaling to numeric columns and save to output_path."""
    df = pd.read_csv(input_path)
    # Keep Invest_Decision if present
    invest_decision = df['Invest_Decision'] if 'Invest_Decision' in df.columns else None
    numeric_cols = [
        'Annual_Revenue_USD', 'Annual_Sales', 'Founder_Investment_USD',
        'Employee_Count', 'Years_Active', 'Revenue_Growth_Rate_%', 'Profit_Margin_%',
        'revenue_per_employee', 'sales_per_employee', 'revenue_per_investment',
        'roi_ratio', 'investment_intensity', 'revenue_maturity', 'sales_to_revenue_ratio'
    ]
    scaler = RobustScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    if invest_decision is not None:
        df['Invest_Decision'] = invest_decision
    df.to_csv(output_path, index=False)
    print(f'Outlier handling and scaling complete. Output: {output_path}')

if __name__ == "__main__":
    handle_outliers()
