import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Pharma Supply Chain ML", layout="wide")

# Header
st.title("🧬 Pharmaceutical Supply Chain Intelligence")
st.markdown("ML-Powered Inventory Optimization for Temperature-Sensitive Biologics")

# Metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast Accuracy", "92.4%", "+5.2%")
col2.metric("Annual Savings", "$1.2M", "18% reduction")
col3.metric("Stockout Rate", "3.1%", "-9.3%")
col4.metric("Inventory Turnover", "6.8x", "Optimal")

# Dummy forecast data for demonstration
forecast_df = {
    'month': ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep'],
    'predicted': [1200, 1300, 1250, 1400, 1350, 1450],
    'actual': [1150, 1280, 1230, 1380, 1320, 1420]
}

# Forecast chart
st.subheader("6-Month Demand Forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=forecast_df['month'], y=forecast_df['predicted'], 
                         name='ML Forecast', line=dict(color='blue', width=3)))
fig.add_trace(go.Scatter(x=forecast_df['month'], y=forecast_df['actual'], 
                         name='Actual', line=dict(color='green', width=3, dash='dash')))
st.plotly_chart(fig, use_container_width=True)

# Add all other sections as needed...
