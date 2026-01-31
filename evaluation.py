# Evaluation: Check for missing values, duplicates, and data types
import pandas as pd

# Evaluation: Check for missing values, duplicates, and data types
import pandas as pd

file_path = 'processed_data_balanced.csv'
df = pd.read_csv(file_path)

print('Data shape:', df.shape)
print('\nMissing values per column:')
print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nData types:')
print(df.dtypes)

# Check for unique values in categorical columns
categorical_cols = ['Sector', 'Stage']
for col in categorical_cols:
    if col in df.columns:
        print(f'\nUnique values in {col}:', df[col].unique())
# investment_model_evaluation_optimization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_score, recall_score, f1_score, roc_auc_score, roc_curve)
import joblib
import json
import warnings

warnings.filterwarnings("ignore")

print("="*60)
print("STARTUP INVESTMENT PREDICTION MODEL")
print("="*60)

# Load dataset
df = pd.read_csv("investor_startup_dataset.csv")
df = pd.read_csv("processed_data_balanced.csv")
print('Data shape:', df.shape)
print('\nMissing values per column:')
print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('\nData types:')
print(df.dtypes)

# Check for unique values in categorical columns
categorical_cols = ['Sector', 'Stage']
for col in categorical_cols:
    if col in df.columns:
        print(f'\nUnique values in {col}:', df[col].unique())

# Prepare features and target
y = df["Invest_Decision"]
# Drop non-numeric, non-categorical columns (like Startup_Name) from features
drop_cols = [col for col in ["Startup_ID", "Invest_Decision", "Startup_Name"] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df["Invest_Decision"]
print(f"\nDataset Info:")
print(f"Total samples: {len(df)}")
print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")
print(f"Target distribution (%):\n{y.value_counts(normalize=True) * 100}")

# Feature groups
categorical_features = ["Sector", "Stage"]
numerical_features = [c for c in X.columns if c not in categorical_features]

print(f"\nNumerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# ============================================
# STEP 1: MODEL SELECTION
# ============================================
print("\n" + "="*60)
print("STEP 1: MODEL SELECTION")
print("="*60)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Define multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    'SVM': SVC(probability=True, random_state=42, class_weight="balanced")
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

selection_results = []

for name, model in models.items():
    print(f"\nEvaluating {name}...")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
    except:
        roc_auc = 0
    
    selection_results.append({
        'Model': name,
        'CV_Accuracy_Mean': cv_scores.mean(),
        'CV_Accuracy_Std': cv_scores.std(),
        'Test_Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc
    })
    
    print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")

selection_df = pd.DataFrame(selection_results).sort_values('Test_Accuracy', ascending=False)
print("\n=== Model Selection Results ===")
print(selection_df.to_string(index=False))

best_model_name = selection_df.iloc[0]['Model']
print(f"\n✓ Best Model Selected: {best_model_name}")

# ============================================
# STEP 2: HYPERPARAMETER OPTIMIZATION
# ============================================
print("\n" + "="*60)
print("STEP 2: HYPERPARAMETER OPTIMIZATION")
print("="*60)

# Define parameter grids
param_grids = {
    'Random Forest': {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'model__n_estimators': [100, 150, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 5, 7]
    },
    'Logistic Regression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l2']
    }
}

# Optimize top 2 models
top_models = selection_df.head(2)['Model'].tolist()
optimized_models = {}
optimization_results = []

for model_name in top_models:
    if model_name in param_grids:
        print(f"\nOptimizing {model_name}...")
        
        # Create base pipeline
        if model_name == 'Random Forest':
            base_model = RandomForestClassifier(random_state=42, class_weight="balanced")
        elif model_name == 'Gradient Boosting':
            base_model = GradientBoostingClassifier(random_state=42)
        elif model_name == 'Logistic Regression':
            base_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("model", base_model)
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        optimized_models[model_name] = best_pipeline
        
        # Evaluate
        y_pred = best_pipeline.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        
        optimization_results.append({
            'Model': model_name,
            'Best_Params': grid_search.best_params_,
            'Best_CV_Score': grid_search.best_score_,
            'Test_Accuracy': test_accuracy,
            'Test_F1': test_f1
        })
        
        print(f"  Best Parameters: {grid_search.best_params_}")
        print(f"  Best CV Score: {grid_search.best_score_:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")

# Select final best model
final_best_model_name = max(optimization_results, key=lambda x: x['Test_Accuracy'])['Model']
final_best_pipeline = optimized_models[final_best_model_name]

print(f"\n✓ Final Best Model: {final_best_model_name}")

# ============================================
# STEP 3: COMPREHENSIVE EVALUATION
# ============================================
print("\n" + "="*60)
print("STEP 3: COMPREHENSIVE EVALUATION")
print("="*60)

# Predictions
y_train_pred = final_best_pipeline.predict(X_train)
y_test_pred = final_best_pipeline.predict(X_test)
y_test_pred_proba = final_best_pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
def calculate_classification_metrics(y_true, y_pred, y_pred_proba, dataset_name):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted') * 100
    recall = recall_score(y_true, y_pred, average='weighted') * 100
    f1 = f1_score(y_true, y_pred, average='weighted') * 100
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except:
        roc_auc = 0
    
    return {
        'Dataset': dataset_name,
        'Accuracy_%': accuracy,
        'Precision_%': precision,
        'Recall_%': recall,
        'F1_Score_%': f1,
        'ROC_AUC': roc_auc
    }

evaluation_metrics = [
    calculate_classification_metrics(y_train, y_train_pred, 
                                    final_best_pipeline.predict_proba(X_train)[:, 1], 'Training'),
    calculate_classification_metrics(y_test, y_test_pred, y_test_pred_proba, 'Test')
]

evaluation_df = pd.DataFrame(evaluation_metrics)
print("\n=== Final Model Performance ===")
print(evaluation_df.to_string(index=False))

test_accuracy = evaluation_df[evaluation_df['Dataset'] == 'Test']['Accuracy_%'].values[0]
test_f1 = evaluation_df[evaluation_df['Dataset'] == 'Test']['F1_Score_%'].values[0]

print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {test_accuracy:.2f}%")
print(f"FINAL TEST F1 SCORE: {test_f1:.2f}%")
print(f"{'='*60}")

# Classification report
print("\n=== Detailed Classification Report ===")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\n=== Confusion Matrix ===")
print(cm)

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Confusion Matrix Heatmap
ax1 = axes[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

# 2. ROC Curve
ax2 = axes[0, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
roc_auc = roc_auc_score(y_test, y_test_pred_proba)
ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

# 3. Feature Importance
ax3 = axes[0, 2]
ohe = final_best_pipeline.named_steps["preprocessing"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(categorical_features))
feature_names = numerical_features + cat_names
importances = final_best_pipeline.named_steps["model"].feature_importances_
indices = np.argsort(importances)[-10:]
ax3.barh(range(len(indices)), importances[indices])
ax3.set_yticks(range(len(indices)))
ax3.set_yticklabels([feature_names[i] for i in indices])
ax3.set_xlabel('Importance')
ax3.set_title('Top 10 Feature Importances')

# 4. Metrics Comparison
ax4 = axes[1, 0]
metrics_df = evaluation_df.set_index('Dataset')[['Accuracy_%', 'Precision_%', 'Recall_%', 'F1_Score_%']]
metrics_df.plot(kind='bar', ax=ax4)
ax4.set_ylabel('Score (%)')
ax4.set_title('Performance Metrics Comparison')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
ax4.legend(loc='lower right')
ax4.grid(axis='y', alpha=0.3)

# 5. Prediction Distribution
ax5 = axes[1, 1]
ax5.hist(y_test_pred_proba, bins=30, edgecolor='black', alpha=0.7)
ax5.axvline(x=0.5, color='r', linestyle='--', lw=2, label='Decision Threshold')
ax5.set_xlabel('Predicted Probability (Class 1)')
ax5.set_ylabel('Frequency')
ax5.set_title('Prediction Probability Distribution')
ax5.legend()
ax5.grid(alpha=0.3)

# 6. Model Comparison
ax6 = axes[1, 2]
model_comparison = selection_df[['Model', 'Test_Accuracy']].sort_values('Test_Accuracy', ascending=True)
ax6.barh(model_comparison['Model'], model_comparison['Test_Accuracy'] * 100)
ax6.set_xlabel('Test Accuracy (%)')
ax6.set_title('Model Selection Comparison')
ax6.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('investment_model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Evaluation visualizations saved")

# Feature importance DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\n=== Top 10 Feature Importances ===")
print(feature_importance_df.head(10).to_string(index=False))

# ============================================
# STEP 4: MODEL DEPLOYMENT
# ============================================
print("\n" + "="*60)
print("STEP 4: MODEL DEPLOYMENT")
print("="*60)

# Save model
model_filename = f'{final_best_model_name.replace(" ", "_")}_investment_model.pkl'
joblib.dump(final_best_pipeline, model_filename)
print(f"✓ Model saved as: {model_filename}")

# Save metadata
deployment_metadata = {
    'model_name': final_best_model_name,
    'model_file': model_filename,
    'test_accuracy': float(test_accuracy),
    'test_f1_score': float(test_f1),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'features': X.columns.tolist(),
    'target_variable': 'Invest_Decision',
    'categorical_features': categorical_features,
    'numerical_features': numerical_features
}

with open('investment_deployment_metadata.json', 'w') as f:
    json.dump(deployment_metadata, f, indent=4)
print("✓ Deployment metadata saved")

# Save report data
report_data = {
    'selection_results': selection_df.to_dict('records'),
    'optimization_results': optimization_results,
    'evaluation_metrics': evaluation_metrics,
    'final_model': final_best_model_name,
    'test_accuracy': float(test_accuracy),
    'test_f1': float(test_f1),
    'feature_importance': feature_importance_df.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'classification_report': classification_report(y_test, y_test_pred, output_dict=True)
}

with open('model_report_data.json', 'w') as f:
    json.dump(report_data, f, indent=4, default=str)
print("✓ Report data saved as model_report_data.json")

print("\n" + "="*60)
print("MODEL PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Final Model: {final_best_model_name}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Test F1 Score: {test_f1:.2f}%")
print(f"Model Status: {'✓ DEPLOYED' if test_accuracy >= 70 else '⚠ NEEDS IMPROVEMENT'}")