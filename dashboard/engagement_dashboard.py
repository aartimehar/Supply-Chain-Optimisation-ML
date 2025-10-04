import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load monthly sales data
df_raw = pd.read_csv('../Data/salesmonthly.csv', header=None)
if df_raw.shape[1] == 1:
	# The file was read as a single column, so split manually
	import csv
	with open('../Data/salesmonthly.csv', 'r') as f:
		reader = csv.reader(f)
		rows = list(reader)
	header = rows[0][0].split(',')
	data = [row[0].split(',') for row in rows[1:]]
	df = pd.DataFrame(data, columns=header)
else:
	df = pd.read_csv('../Data/salesmonthly.csv')
df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '').str.lower()
if 'datum' in df.columns:
	df = df.rename(columns={'datum': 'date'})
elif 'date' not in df.columns:
	st.error("Could not find a 'date' or 'datum' column in the sales data. Please check the CSV format.")
	st.stop()
df['date'] = pd.to_datetime(df['date'])
df.columns = df.columns.str.strip().str.lower()
if 'datum' in df.columns:
	df = df.rename(columns={'datum': 'date'})
elif 'date' not in df.columns:
	# Try to remove quotes from column names if present
	df.columns = df.columns.str.replace('"', '')
	if 'datum' in df.columns:
		df = df.rename(columns={'datum': 'date'})
	elif 'date' not in df.columns:
		st.error("Could not find a 'date' or 'datum' column in the sales data. Please check the CSV format.")
		st.stop()
df['date'] = pd.to_datetime(df['date'])

# Melt data for product-wise analysis
df_long = df.melt(id_vars='date', var_name='product', value_name='sales')

# Sidebar filters
st.sidebar.header('Filter Data')
products = st.sidebar.multiselect('Select Product(s)', options=df_long['product'].unique(), default=list(df_long['product'].unique()))
date_range = st.sidebar.date_input('Select Date Range', [df_long['date'].min(), df_long['date'].max()])

filtered = df_long[(df_long['product'].isin(products)) & (df_long['date'] >= pd.to_datetime(date_range[0])) & (df_long['date'] <= pd.to_datetime(date_range[1]))]

st.set_page_config(page_title="Engagement & Sales Dashboard", layout="wide")
st.title("📊 Biologics Supply Chain & User Engagement Dashboard")
st.markdown("Key metrics and trends for marketing and sales optimization.")

# Metrics row (dummy values for illustration)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Monthly Sales Growth", "8.2%", "+1.1%")
col2.metric("Avg Deal Size", "$12,500", "+$500")
col3.metric("Lead Conversion Rate", "14.7%", "+2.3%")
col4.metric("Sales Funnel Progress", "67%", "On Track")
col5.metric("Top Region", "EMEA", "↑")

# Line graph: Sales trends over time
st.subheader("Sales Trends Over Time")
fig1 = px.line(filtered, x='date', y='sales', color='product', title='Monthly Sales by Product')
st.plotly_chart(fig1, use_container_width=True)

# Bar chart: Compare product sales
st.subheader("Product Sales Comparison")
latest_month = filtered['date'].max()
latest_sales = filtered[filtered['date'] == latest_month]
fig2 = px.bar(latest_sales, x='product', y='sales', title=f'Sales by Product ({latest_month.date()})')
st.plotly_chart(fig2, use_container_width=True)

# Heat map: Dummy user activity data
st.subheader("User Activity Heatmap")
import numpy as np
activity = np.random.rand(7, 24)  # 7 days, 24 hours
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
hours = [f'{h}:00' for h in range(24)]
fig3 = go.Figure(data=go.Heatmap(z=activity, x=hours, y=days, colorscale='Viridis'))
fig3.update_layout(title='Website User Activity (Demo)')
st.plotly_chart(fig3, use_container_width=True)

# Drill-down: Show details for selected product
st.subheader("Drill-Down: Product Details")
selected_product = st.selectbox('Choose a product for details', options=products)
product_data = filtered[filtered['product'] == selected_product]
st.write(product_data)

# Layout and design notes
st.markdown("""
- **Filtering:** Use sidebar to filter by product and date range.
- **Drill-down:** Select a product for detailed data.
- **Visuals:** Line, bar, and heatmap for clarity.
- **Design:** Clean layout, clear labels, limited color palette.
- **Data Pipeline:** Connect to source systems for real-time updates.
- **Integrity:** Regular system checks recommended.
""")
