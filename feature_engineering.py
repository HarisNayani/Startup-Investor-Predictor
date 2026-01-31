
import pandas as pd

def feature_engineering(input_path='investor_startup_dataset_large.csv', output_path='processed_data.csv'):
	"""Add domain-specific features to the dataset and save to output_path."""
	df = pd.read_csv(input_path)
	# If Invest_Decision exists, keep it; else, fill with NaN
	if 'Invest_Decision' not in df.columns:
		df['Invest_Decision'] = pd.NA
	# Feature Engineering
	df['revenue_per_employee'] = df['Annual_Revenue_USD'] / df['Employee_Count'].replace(0, 1)
	df['sales_per_employee'] = df['Annual_Sales'] / df['Employee_Count'].replace(0, 1)
	df['revenue_per_investment'] = df['Annual_Revenue_USD'] / df['Founder_Investment_USD'].replace(0, 1)
	df['roi_ratio'] = df['Profit_Margin_%'] / (df['Founder_Investment_USD'].replace(0, 1))
	df['investment_intensity'] = df['Founder_Investment_USD'] / df['Employee_Count'].replace(0, 1)
	df['revenue_maturity'] = df['Annual_Revenue_USD'] * df['Years_Active']
	df['is_profitable'] = (df['Profit_Margin_%'] > 0).astype(int)
	df['sales_to_revenue_ratio'] = df['Annual_Sales'] / (df['Annual_Revenue_USD'].replace(0, 1))
	df.to_csv(output_path, index=False)
	print(f'Feature engineering complete. Output: {output_path}')

if __name__ == "__main__":
	feature_engineering()
