# model_training.py - Train and evaluate models for Revenue Growth Rate prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Load the preprocessed data
X_train = pd.read_csv('X_train_scaled.csv')
X_val = pd.read_csv('X_val_scaled.csv')
X_test = pd.read_csv('X_test_scaled.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()
y_val = pd.read_csv('y_val.csv').values.ravel()
y_test = pd.read_csv('y_test.csv').values.ravel()

print("=== Data Loaded Successfully ===")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
    'Support Vector Machine': SVR(kernel='rbf', C=1.0)
}

# Store results
results = {
    'Model': [],
    'Train_R2': [],
    'Val_R2': [],
    'Test_R2': [],
    'Train_RMSE': [],
    'Val_RMSE': [],
    'Test_RMSE': [],
    'Train_MAE': [],
    'Val_MAE': [],
    'Test_MAE': []
}

trained_models = {}

print("\n=== Training Models ===\n")

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Store results
    results['Model'].append(name)
    results['Train_R2'].append(train_r2)
    results['Val_R2'].append(val_r2)
    results['Test_R2'].append(test_r2)
    results['Train_RMSE'].append(train_rmse)
    results['Val_RMSE'].append(val_rmse)
    results['Test_RMSE'].append(test_rmse)
    results['Train_MAE'].append(train_mae)
    results['Val_MAE'].append(val_mae)
    results['Test_MAE'].append(test_mae)
    
    # Save trained model
    trained_models[name] = model
    
    print(f"  Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Test RMSE: {test_rmse:.4f}\n")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Val_R2', ascending=False)

print("\n=== Model Performance Summary ===")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('model_performance_results.csv', index=False)
print("\nResults saved to 'model_performance_results.csv'")

# Visualize model performance
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# R² Score comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(results_df))
width = 0.25
ax1.bar(x_pos - width, results_df['Train_R2'], width, label='Train', alpha=0.8)
ax1.bar(x_pos, results_df['Val_R2'], width, label='Validation', alpha=0.8)
ax1.bar(x_pos + width, results_df['Test_R2'], width, label='Test', alpha=0.8)
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score')
ax1.set_title('R² Score Comparison Across Models')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# RMSE comparison
ax2 = axes[0, 1]
ax2.bar(x_pos - width, results_df['Train_RMSE'], width, label='Train', alpha=0.8)
ax2.bar(x_pos, results_df['Val_RMSE'], width, label='Validation', alpha=0.8)
ax2.bar(x_pos + width, results_df['Test_RMSE'], width, label='Test', alpha=0.8)
ax2.set_xlabel('Models')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE Comparison Across Models')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# MAE comparison
ax3 = axes[1, 0]
ax3.bar(x_pos - width, results_df['Train_MAE'], width, label='Train', alpha=0.8)
ax3.bar(x_pos, results_df['Val_MAE'], width, label='Validation', alpha=0.8)
ax3.bar(x_pos + width, results_df['Test_MAE'], width, label='Test', alpha=0.8)
ax3.set_xlabel('Models')
ax3.set_ylabel('MAE')
ax3.set_title('MAE Comparison Across Models')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Overfitting check (Train vs Val R²)
ax4 = axes[1, 1]
ax4.scatter(results_df['Train_R2'], results_df['Val_R2'], s=100, alpha=0.6)
for i, model in enumerate(results_df['Model']):
    ax4.annotate(model, (results_df['Train_R2'].iloc[i], results_df['Val_R2'].iloc[i]), 
                fontsize=8, ha='right')
ax4.plot([0, 1], [0, 1], 'r--', label='Perfect fit')
ax4.set_xlabel('Training R²')
ax4.set_ylabel('Validation R²')
ax4.set_title('Overfitting Check: Train vs Validation R²')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("Performance comparison plot saved to 'model_performance_comparison.png'")

# Get best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

print(f"\n=== Best Model: {best_model_name} ===")
print(f"Validation R²: {results_df.iloc[0]['Val_R2']:.4f}")
print(f"Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
print(f"Test RMSE: {results_df.iloc[0]['Test_RMSE']:.4f}")
print(f"Test MAE: {results_df.iloc[0]['Test_MAE']:.4f}")

# Feature importance (for tree-based models)
if best_model_name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n=== Top 10 Important Features ({best_model_name}) ===")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'].head(10), feature_importance['Importance'].head(10))
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance plot saved to 'feature_importance.png'")

# Prediction vs Actual plot for best model
y_test_pred_best = best_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Revenue Growth Rate (%)')
plt.ylabel('Predicted Revenue Growth Rate (%)')
plt.title(f'Actual vs Predicted - {best_model_name}')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("Actual vs Predicted plot saved to 'actual_vs_predicted.png'")

print("\n=== Model Training Complete! ===")