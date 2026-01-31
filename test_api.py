import requests

# Example data (replace values as needed)
data = {
    "Annual_Revenue_USD": 100000,
    "Annual_Sales": 80000,
    "Founder_Investment_USD": 20000,
    "Employee_Count": 10,
    "Years_Active": 2,
    "Revenue_Growth_Rate_%": 15,
    "Profit_Margin_%": 10,
    "revenue_per_employee": 10000,
    "sales_per_employee": 8000,
    "revenue_per_investment": 5,
    "roi_ratio": 0.5,
    "investment_intensity": 0.2,
    "revenue_maturity": 1,
    "is_profitable": 1,
    "sales_to_revenue_ratio": 0.8,
    "Sector": "Energy",
    "Stage": "Series A"
}

response = requests.post("http://localhost:5000/predict", json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
