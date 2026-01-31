import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# load dataset
data=pd.read_csv('investor_startup_dataset.csv')

# display first few rows
print(data.head())

# identifying and printing numerical and categorical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("Numerical Columns:", numerical_columns)
print("Categorical Columns:", categorical_columns)

# box plots for numerical and categorical columns
for cols in numerical_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data[cols])
    plt.title(f'Distribution of {cols}')
    # plt.show()

for cols in categorical_columns:
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=data[cols])
    plt.title(f'Distribution of {cols}')
    # plt.show()

# Data preprocessing
# handling missing values
print("Missing values in each column:")
print(data.isnull().sum())
print("Shape before dropping missing values:", data.shape)
data = data.dropna()
print("After dropping missing values, new shape:", data.shape)


# Feature Engineering and Selection

# 1. Check data types and basic statistics
print("\n=== Data Info ===")
print(data.info())
print("\n=== Statistical Summary ===")
print(data.describe())

# 2. Encode categorical variables
# Create a copy to preserve original data
data_encoded = data.copy()

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data_encoded[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"\nEncoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Correlation analysis
print("\n=== Correlation Matrix ===")
correlation_matrix = data_encoded.corr()
print(correlation_matrix)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
# plt.show()

# 4. Feature importance (if you have a target variable)
# Identify your target variable first
# target = 'your_target_column'  # Replace with actual target
# X = data_encoded.drop(target, axis=1)
# y = data_encoded[target]

# 5. Check for multicollinearity
print("\n=== High Correlations (>0.7) ===")
high_corr = correlation_matrix.abs() > 0.7
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if high_corr.iloc[i, j]:
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                   correlation_matrix.columns[j], 
                                   correlation_matrix.iloc[i, j]))
for pair in high_corr_pairs:
    print(f"{pair[0]} <-> {pair[1]}: {pair[2]:.3f}")

# 6. Create new features based on startup/investor patterns

# Revenue efficiency metrics
data_encoded['revenue_per_employee'] = data_encoded['Annual_Revenue_USD'] / (data_encoded['Employee_Count'] + 1)
data_encoded['sales_per_employee'] = data_encoded['Annual_Sales'] / (data_encoded['Employee_Count'] + 1)

# Investment efficiency
data_encoded['revenue_per_investment'] = data_encoded['Annual_Revenue_USD'] / (data_encoded['Founder_Investment_USD'] + 1)
data_encoded['roi_ratio'] = (data_encoded['Annual_Revenue_USD'] * data_encoded['Profit_Margin_%'] / 100) / (data_encoded['Founder_Investment_USD'] + 1)

# Company maturity indicator
data_encoded['investment_intensity'] = data_encoded['Founder_Investment_USD'] / (data_encoded['Years_Active'] + 1)
data_encoded['revenue_maturity'] = data_encoded['Annual_Revenue_USD'] / (data_encoded['Years_Active'] + 1)

# Profitability indicator
data_encoded['is_profitable'] = (data_encoded['Profit_Margin_%'] > 0).astype(int)

# Sales to revenue ratio (efficiency metric)
data_encoded['sales_to_revenue_ratio'] = data_encoded['Annual_Sales'] / (data_encoded['Annual_Revenue_USD'] + 1)

print("\n=== New Features Created ===")
new_features = ['revenue_per_employee', 'sales_per_employee', 'revenue_per_investment', 
                'roi_ratio', 'investment_intensity', 'revenue_maturity', 
                'is_profitable', 'sales_to_revenue_ratio']
print(new_features)

# Display correlation of new features with target
print("\n=== Correlation with Revenue_Growth_Rate ===")
target_correlations = data_encoded[new_features + ['Revenue_Growth_Rate_%']].corr()['Revenue_Growth_Rate_%'].sort_values(ascending=False)
print(target_correlations)

# 7. Scale numerical features (for later modeling)

scaler = StandardScaler()
numerical_features_scaled = pd.DataFrame(
    scaler.fit_transform(data_encoded[numerical_columns]),
    columns=numerical_columns,
    index=data_encoded.index
)

print("\n=== Scaled Features Summary ===")
print(numerical_features_scaled.describe())

# 8. Save processed data
data_encoded.to_csv('processed_data.csv', index=False)
print("\nProcessed data saved to 'processed_data.csv'")
# Data Splitting for Modeling

# Define target and features
target = 'Revenue_Growth_Rate_%'
X = data_encoded.drop(target, axis=1)
y = data_encoded[target]

print(f"\n=== Target Variable: {target} ===")
print(f"Target statistics:")
print(y.describe())

# Split data: 70% training, 15% validation, 15% test
# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)

# Second split: Split temp into validation and test (50-50 of the 30%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
)

print(f"\n=== Data Split Summary ===")
print(f"Total samples: {len(data_encoded)}")
print(f"Training set: {len(X_train)} samples ({len(X_train)/len(data_encoded)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(data_encoded)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(data_encoded)*100:.1f}%)")

print(f"\n=== Target Distribution Across Splits ===")
print(f"Training - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"Validation - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}")
print(f"Test - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

# Visualize target distribution across splits
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(y_train, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title('Training Set Distribution')
axes[0].set_xlabel(target)

axes[1].hist(y_val, bins=30, edgecolor='black', alpha=0.7)
axes[1].set_title('Validation Set Distribution')
axes[1].set_xlabel(target)

axes[2].hist(y_test, bins=30, edgecolor='black', alpha=0.7)
axes[2].set_title('Test Set Distribution')
axes[2].set_xlabel(target)

plt.tight_layout()
plt.savefig('target_distribution_splits.png', dpi=300, bbox_inches='tight')
# plt.show()

# Scale features for modeling

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print("\n=== Data Ready for Modeling ===")
print(f"Feature columns: {X_train.shape[1]}")
print(f"Features: {list(X_train.columns)}")

# Save the processed datasets
X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_val_scaled.to_csv('X_val_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_val.to_csv('y_val.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("\nScaled datasets saved for modeling!")