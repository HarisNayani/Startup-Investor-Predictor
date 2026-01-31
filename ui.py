import streamlit as st
import requests

st.title("Startup Investment Predictor")
st.write("Enter startup details below to get an investment decision.")

# Define input fields
def user_form():
    col1, col2 = st.columns(2)
    with col1:
        annual_revenue = st.number_input("Annual Revenue (USD)", value=100000)
        annual_sales = st.number_input("Annual Sales", value=80000)
        founder_investment = st.number_input("Founder Investment (USD)", value=20000)
        employee_count = st.number_input("Employee Count", value=10)
        years_active = st.number_input("Years Active", value=2)
        revenue_growth = st.number_input("Revenue Growth Rate (%)", value=15)
        profit_margin = st.number_input("Profit Margin (%)", value=10)
        revenue_per_employee = st.number_input("Revenue per Employee", value=10000)
        sales_per_employee = st.number_input("Sales per Employee", value=8000)
    with col2:
        revenue_per_investment = st.number_input("Revenue per Investment", value=5)
        roi_ratio = st.number_input("ROI Ratio", value=0.5)
        investment_intensity = st.number_input("Investment Intensity", value=0.2)
        revenue_maturity = st.number_input("Revenue Maturity", value=1)
        is_profitable = st.selectbox("Is Profitable", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        sales_to_revenue_ratio = st.number_input("Sales to Revenue Ratio", value=0.8)
        sector = st.text_input("Sector", value="Energy")
        stage = st.text_input("Stage", value="Series A")
    return {
        "Annual_Revenue_USD": annual_revenue,
        "Annual_Sales": annual_sales,
        "Founder_Investment_USD": founder_investment,
        "Employee_Count": employee_count,
        "Years_Active": years_active,
        "Revenue_Growth_Rate_%": revenue_growth,
        "Profit_Margin_%": profit_margin,
        "revenue_per_employee": revenue_per_employee,
        "sales_per_employee": sales_per_employee,
        "revenue_per_investment": revenue_per_investment,
        "roi_ratio": roi_ratio,
        "investment_intensity": investment_intensity,
        "revenue_maturity": revenue_maturity,
        "is_profitable": is_profitable,
        "sales_to_revenue_ratio": sales_to_revenue_ratio,
        "Sector": sector,
        "Stage": stage
    }

user_input = user_form()

if st.button("Predict Investment Decision"):
    with st.spinner("Predicting..."):
        try:
            response = requests.post("http://localhost:5000/predict", json=user_input)
            result = response.json()
            if response.status_code == 200:
                st.success(f"Decision: {result['decision']} (Confidence: {result['confidence']*100:.2f}%)")
                st.info(f"Raw Prediction: {result['prediction']}")
                st.write("Check your terminal for model accuracy.")
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"Request failed: {e}")
