# generate_investment_report.py
import pandas as pd
import json
from datetime import datetime

# Load report data
with open('model_report_data.json', 'r') as f:
    report_data = json.load(f)

test_accuracy = report_data['test_accuracy']
test_f1 = report_data['test_f1']
final_model = report_data['final_model']
cm = report_data['confusion_matrix']
class_report = report_data['classification_report']

# Generate HTML Report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Investment Decision Model Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #1e3c72;
            border-bottom: 3px solid #1e3c72;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        .section h3 {{
            color: #2a5298;
            margin-top: 20px;
        }}
        .metric-card {{
            display: inline-block;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 20px;
            margin: 10px;
            border-radius: 8px;
            min-width: 200px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-card h3 {{
            margin: 0;
            font-size: 2.5em;
            color: white;
        }}
        .metric-card p {{
            margin: 5px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #1e3c72;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .success {{
            color: #28a745;
            font-weight: bold;
        }}
        .warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .danger {{
            color: #dc3545;
            font-weight: bold;
        }}
        .alert {{
            padding: 15px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 5px solid;
        }}
        .alert-success {{
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }}
        ul {{
            line-height: 2;
        }}
        .code {{
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
        .cm-table {{
            max-width: 400px;
            margin: 20px auto;
        }}
        .cm-table td {{
            text-align: center;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Investment Decision Model Report</h1>
        <p>Startup Investment Prediction System</p>
        <p>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <div style="text-align: center;">
            <div class="metric-card">
                <h3>{test_accuracy:.2f}%</h3>
                <p>Test Accuracy</p>
            </div>
            <div class="metric-card">
                <h3>{test_f1:.2f}%</h3>
                <p>F1 Score</p>
            </div>
            <div class="metric-card">
                <h3>{final_model}</h3>
                <p>Best Model</p>
            </div>
        </div>
        
        {'<div class="alert alert-success">&#10003; Model performance is EXCELLENT (Accuracy &ge; 80%)</div>' if test_accuracy >= 80 else ''}
        {'<div class="alert alert-warning">&#9888; Model performance is GOOD but has room for improvement (70% &le; Accuracy &lt; 80%)</div>' if 70 <= test_accuracy < 80 else ''}
        {'<div class="alert alert-danger">&#10007; Model performance is BELOW TARGET (Accuracy &lt; 70%) - Improvement needed</div>' if test_accuracy < 70 else ''}
    </div>

    <div class="section">
        <h2>Project Objective</h2>
        <p>To develop a machine learning classification model that predicts whether to <strong>invest</strong> or <strong>not invest</strong> in startups 
        based on key business metrics including sector, stage, revenue, employee count, and other financial indicators.</p>
        
        <p><strong>Business Value:</strong></p>
        <ul>
            <li>Automate preliminary investment screening</li>
            <li>Reduce time spent on non-viable opportunities</li>
            <li>Identify high-potential startups for deeper due diligence</li>
            <li>Data-driven decision support for investment committees</li>
        </ul>
    </div>

    <div class="section">
        <h2>Model Development Pipeline</h2>
        
        <h3>1. Model Selection</h3>
        <p>We evaluated <strong>{len(report_data['selection_results'])}</strong> different classification algorithms:</p>
        <table>
            <tr>
                <th>Model</th>
                <th>CV Accuracy</th>
                <th>Test Accuracy</th>
                <th>F1 Score</th>
                <th>ROC-AUC</th>
            </tr>
"""

for result in report_data['selection_results']:
    html_report += f"""
            <tr>
                <td>{result['Model']}</td>
                <td>{result['CV_Accuracy_Mean']:.4f} &plusmn; {result['CV_Accuracy_Std']:.4f}</td>
                <td>{result['Test_Accuracy']:.4f}</td>
                <td>{result['F1_Score']:.4f}</td>
                <td>{result['ROC_AUC']:.4f}</td>
            </tr>
"""

html_report += f"""
        </table>
        <p><strong>Selected Model:</strong> {final_model} demonstrated the best overall performance.</p>

        <h3>2. Hyperparameter Optimization</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Best Parameters</th>
                <th>CV Score</th>
                <th>Test Accuracy</th>
            </tr>
"""

for result in report_data['optimization_results']:
    params_str = '<br>'.join([f"{k.replace('model__', '')}: {v}" for k, v in result['Best_Params'].items()])
    html_report += f"""
            <tr>
                <td>{result['Model']}</td>
                <td style="font-size: 0.85em;">{params_str}</td>
                <td>{result['Best_CV_Score']:.4f}</td>
                <td>{result['Test_Accuracy']:.4f}</td>
            </tr>
"""

html_report += """
        </table>

        <h3>3. Final Model Evaluation</h3>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Accuracy (%)</th>
                <th>Precision (%)</th>
                <th>Recall (%)</th>
                <th>F1 Score (%)</th>
            </tr>
"""

for metric in report_data['evaluation_metrics']:
    accuracy_class = 'success' if metric['Accuracy_%'] >= 80 else ('warning' if metric['Accuracy_%'] >= 70 else 'danger')
    html_report += f"""
            <tr>
                <td><strong>{metric['Dataset']}</strong></td>
                <td class="{accuracy_class}">{metric['Accuracy_%']:.2f}%</td>
                <td>{metric['Precision_%']:.2f}%</td>
                <td>{metric['Recall_%']:.2f}%</td>
                <td>{metric['F1_Score_%']:.2f}%</td>
            </tr>
"""

html_report += f"""
        </table>
    </div>

    <div class="section">
        <h2>Confusion Matrix</h2>
        <p>Performance breakdown by prediction type:</p>
        <table class="cm-table">
            <tr>
                <th></th>
                <th>Predicted: No</th>
                <th>Predicted: Yes</th>
            </tr>
            <tr>
                <th>Actual: No</th>
                <td>{cm[0][0]}</td>
                <td style="color: #dc3545;">{cm[0][1]}</td>
            </tr>
            <tr>
                <th>Actual: Yes</th>
                <td style="color: #dc3545;">{cm[1][0]}</td>
                <td>{cm[1][1]}</td>
            </tr>
        </table>
        
        <p><strong>Key Metrics:</strong></p>
        <ul>
            <li><strong>True Negatives (Correct "Don't Invest"):</strong> {cm[0][0]}</li>
            <li><strong>True Positives (Correct "Invest"):</strong> {cm[1][1]}</li>
            <li><strong>False Positives (Missed Bad Investment):</strong> {cm[0][1]} - Type I Error</li>
            <li><strong>False Negatives (Missed Good Investment):</strong> {cm[1][0]} - Type II Error</li>
        </ul>
    </div>

    <div class="section">
        <h2>Detailed Classification Report</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
"""

for class_name, metrics in class_report.items():
    if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
        html_report += f"""
            <tr>
                <td><strong>Class {class_name}</strong></td>
                <td>{metrics['precision']:.4f}</td>
                <td>{metrics['recall']:.4f}</td>
                <td>{metrics['f1-score']:.4f}</td>
                <td>{int(metrics['support'])}</td>
            </tr>
"""

html_report += """
        </table>
    </div>

    <div class="section">
        <h2>Feature Importance Analysis</h2>
        <p>Top features driving investment predictions:</p>
        <table>
            <tr>
                <th>Rank</th>
                <th>Feature</th>
                <th>Importance Score</th>
            </tr>
"""

for idx, feat in enumerate(report_data['feature_importance'][:10], 1):
    html_report += f"""
            <tr>
                <td>{idx}</td>
                <td>{feat['Feature']}</td>
                <td>{feat['Importance']:.4f}</td>
            </tr>
"""

html_report += """
        </table>
    </div>
"""

# Issues and Solutions (if accuracy < 80%)
if test_accuracy < 80:
    html_report += f"""
    <div class="section">
        <h2>Performance Issues &amp; Improvement Strategies</h2>
        
        <div class="alert alert-warning">
            <strong>Current Status:</strong> The model achieved {test_accuracy:.2f}% accuracy, which is below the 80% target.
        </div>

        <h3>Identified Issues:</h3>
        <ul>
"""
    
    issues = []
    if test_accuracy < 70:
        issues.append("Accuracy significantly below acceptable threshold")
    
    if cm[0][1] > cm[1][0]:
        issues.append("High False Positive rate - model is recommending bad investments")
    elif cm[1][0] > cm[0][1]:
        issues.append("High False Negative rate - model is missing good investment opportunities")
    
    if test_f1 < 70:
        issues.append("Low F1 score indicates imbalanced precision/recall")
    
    # Check overfitting
    train_metrics = next(m for m in report_data['evaluation_metrics'] if m['Dataset'] == 'Training')
    test_metrics = next(m for m in report_data['evaluation_metrics'] if m['Dataset'] == 'Test')
    
    if train_metrics['Accuracy_%'] - test_metrics['Accuracy_%'] > 10:
        issues.append("Model overfitting detected (large train-test performance gap)")
    
    for issue in issues:
        html_report += f"            <li>{issue}</li>\n"
    
    html_report += """
        </ul>

        <h3>Recommended Solutions:</h3>
        
        <h4>1. Data Improvements</h4>
        <ul>
            <li>Collect more training examples (target 1000+ samples)</li>
            <li>Balance dataset classes if imbalanced</li>
            <li>Add more discriminative features (market trends, competitor analysis)</li>
            <li>Feature engineering: interaction terms, ratios, growth rates</li>
        </ul>

        <h4>2. Model Improvements</h4>
        <ul>
            <li>Try XGBoost or LightGBM models</li>
            <li>Ensemble multiple models (voting or stacking)</li>
            <li>Tune decision threshold based on business costs</li>
            <li>Apply SMOTE for class imbalance</li>
        </ul>

        <h4>3. Feature Engineering</h4>
        <ul>
            <li>Create domain-specific ratios (revenue/employee, growth velocity)</li>
            <li>Add temporal features (years since founding, funding velocity)</li>
            <li>Industry benchmarking features</li>
            <li>Remove low-importance features</li>
        </ul>

        <div class="code">
            <strong>Quick Win - Try XGBoost:</strong><br><br>
            # pip install xgboost --break-system-packages<br>
            from xgboost import XGBClassifier<br>
            <br>
            model = XGBClassifier(<br>
            &nbsp;&nbsp;&nbsp;&nbsp;n_estimators=200,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;learning_rate=0.05,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;max_depth=6,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;scale_pos_weight=1,<br>
            &nbsp;&nbsp;&nbsp;&nbsp;random_state=42<br>
            )<br>
            # Often achieves 5-10% better accuracy!
        </div>
    </div>
"""

html_report += f"""
    <div class="section">
        <h2>Deployment Information</h2>
        
        <h3>Model Artifacts</h3>
        <ul>
            <li><strong>Model File:</strong> {final_model.replace(' ', '_')}_investment_model.pkl</li>
            <li><strong>Model Type:</strong> {final_model}</li>
            <li><strong>Status:</strong> <span class="{'success' if test_accuracy >= 70 else 'warning'}">
                {'&#10003; PRODUCTION READY' if test_accuracy >= 70 else '&#9888; NEEDS IMPROVEMENT'}</span></li>
        </ul>

        <h3>Usage Example</h3>
        <div class="code">
            import joblib<br>
            import pandas as pd<br>
            <br>
            # Load model<br>
            model = joblib.load('{final_model.replace(' ', '_')}_investment_model.pkl')<br>
            <br>
            # Prepare new startup data<br>
            new_startup = pd.DataFrame({{<br>
            &nbsp;&nbsp;&nbsp;&nbsp;'Sector': ['Technology'],<br>
            &nbsp;&nbsp;&nbsp;&nbsp;'Stage': ['Series A'],<br>
            &nbsp;&nbsp;&nbsp;&nbsp;'Annual_Revenue_USD': [500000],<br>
            &nbsp;&nbsp;&nbsp;&nbsp;# ... other features<br>
            }})<br>
            <br>
            # Make prediction<br>
            prediction = model.predict(new_startup)[0]<br>
            probability = model.predict_proba(new_startup)[0, 1]<br>
            <br>
            print(f"Decision: {{'Invest' if prediction == 1 else 'Don\\'t Invest'}}")<br>
            print(f"Confidence: {{probability:.2%}}")
        </div>
    </div>

    <div class="section">
        <h2>Conclusions &amp; Next Steps</h2>
        <p><strong>Model Performance:</strong> The {final_model} achieved {test_accuracy:.2f}% accuracy in predicting investment decisions.</p>
        
        <p><strong>Business Impact:</strong></p>
        <ul>
            <li>Automated screening can process {int(cm[0][0] + cm[1][1])} out of {int(sum(sum(cm, [])))} startups correctly</li>
            <li>Reduces manual review workload by approximately {test_accuracy:.0f}%</li>
            <li>Prioritizes {cm[1][1]} high-potential opportunities for deeper analysis</li>
        </ul>

        <p><strong>Recommended Actions:</strong></p>
        <ol>
            {'<li>Deploy model to production for automated screening</li>' if test_accuracy >= 70 else '<li>Implement improvements before deployment</li>'}
            <li>Set up monitoring dashboard to track prediction accuracy</li>
            <li>Collect feedback on investment outcomes to retrain model</li>
            <li>Quarterly model updates with new startup data</li>
            <li>A/B test model recommendations against human decisions</li>
        </ol>
    </div>

    <div class="footer">
        <p>Investment Model Report | &copy; 2026</p>
        <p>For questions, contact the Data Science Team</p>
    </div>
</body>
</html>
"""

# Save report
with open('Investment_Model_Report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)

print("="*60)
print("INVESTMENT MODEL REPORT GENERATED")
print("="*60)
print(f"Report saved as: Investment_Model_Report.html")
print(f"Model Accuracy: {test_accuracy:.2f}%")
print(f"Model F1 Score: {test_f1:.2f}%")
print("="*60)