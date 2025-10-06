import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Template and validation functions
def create_csv_template():
    """Generate a CSV template for users to follow"""
    template_data = {
        'date': ['2024-01-31', '2024-02-28', '2024-03-31', '2024-04-30', '2024-05-31'],
        'product_a': [1000, 1200, 1100, 1300, 1150],
        'product_b': [800, 900, 850, 950, 875],
        'product_c': [600, 700, 650, 750, 680],
        'product_d': [400, 500, 450, 550, 475]
    }
    return pd.DataFrame(template_data)

def validate_uploaded_data(df):
    """Validate uploaded CSV format"""
    issues = []
    
    # Check for date column
    date_columns = ['date', 'datum', 'Date', 'DATE']
    if not any(col in df.columns for col in date_columns):
        issues.append("❌ Missing date column (expected: 'date')")
    
    # Check for at least 2 product columns
    non_date_cols = [col for col in df.columns if col.lower() not in ['date', 'datum']]
    if len(non_date_cols) < 2:
        issues.append("❌ Need at least 2 product columns")
    
    # Check data types
    try:
        df_test = df.copy()
        for date_col in date_columns:
            if date_col in df_test.columns:
                pd.to_datetime(df_test[date_col])
                break
    except:
        issues.append("❌ Date column format invalid (use YYYY-MM-DD)")
    
    return issues

def process_uploaded_data(uploaded_file):
    """Process and clean uploaded CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate format
        validation_issues = validate_uploaded_data(df)
        if validation_issues:
            st.error("📋 Data validation failed:")
            for issue in validation_issues:
                st.write(issue)
            st.info("💡 Please download the template and follow the format.")
            return None
        
        # Clean and standardize
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle date column
        date_columns = ['date', 'datum']
        for date_col in date_columns:
            if date_col in df.columns:
                df['date'] = pd.to_datetime(df[date_col])
                if date_col != 'date':
                    df = df.drop(columns=[date_col])
                break
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col != 'date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="Advanced Sales Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and process data
@st.cache_data
def load_data(data_source="default", uploaded_file=None):
    # Handle uploaded data first
    if data_source == "uploaded" and uploaded_file is not None:
        df = process_uploaded_data(uploaded_file)
        if df is not None:
            return df, "uploaded"
        else:
            st.warning("⚠️ Failed to load uploaded data. Using default data instead.")
    
    # Default data loading (your pharmaceutical data)
    try:
        # Try multiple possible paths for the CSV file
        possible_paths = [
            'Data/salesmonthly.csv',           # Running from root
            '../Data/salesmonthly.csv',        # Running from dashboard folder
            'C:/Users/aarti/Documents/Supply Chain Optimisation ML/Data/salesmonthly.csv'  # Absolute path
        ]
        
        df = None
        for path in possible_paths:
            try:
                # Try to read the CSV file
                df_raw = pd.read_csv(path, header=None)
                if df_raw.shape[1] == 1:
                    import csv
                    with open(path, 'r') as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                    header = rows[0][0].split(',')
                    data = [row[0].split(',') for row in rows[1:]]
                    df = pd.DataFrame(data, columns=header)
                else:
                    df = pd.read_csv(path)
                break  # If successful, break out of the loop
            except FileNotFoundError:
                continue  # Try the next path
        
        if df is None:
            raise FileNotFoundError("Could not find salesmonthly.csv in any expected location")
        
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '').str.lower()
        if 'datum' in df.columns:
            df = df.rename(columns={'datum': 'date'})
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        numeric_cols = [col for col in df.columns if col != 'date']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, "default"
    except Exception as e:
        st.warning(f"Could not load data file: {str(e)}. Using sample data.")
        # Return sample data
        dates = pd.date_range('2020-01-01', periods=48, freq='M')
        dummy_data = {
            'date': dates,
            'product_a': np.random.randint(500, 2000, 48),
            'product_b': np.random.randint(300, 1500, 48),
            'product_c': np.random.randint(400, 1800, 48),
            'product_d': np.random.randint(200, 1200, 48)
        }
        return pd.DataFrame(dummy_data), "sample"

# Sidebar - Data Source Selection
st.sidebar.title("🎯 Sales Dashboard Navigation")

# Add data source options
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Source")

# Template download
template_df = create_csv_template()
csv_template = template_df.to_csv(index=False)
st.sidebar.download_button(
    label="📥 Download CSV Template",
    data=csv_template,
    file_name="sales_data_template.csv",
    mime="text/csv",
    help="Download template and fill with your sales data"
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Your Sales Data (Optional)",
    type=['csv'],
    help="Upload CSV following the template format"
)

# Data source selection
if uploaded_file is not None:
    data_source = "uploaded"
    st.sidebar.success("✅ Custom data uploaded!")
    st.sidebar.write(f"**File:** {uploaded_file.name}")
else:
    data_source = "default"
    st.sidebar.info("📈 Showing pharmaceutical sales data")

# Load data based on source
df, data_type = load_data(data_source, uploaded_file)
# Show data preview and instructions
if data_type == "uploaded":
    # Show uploaded data preview
    st.subheader("📊 Your Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Points", len(df))
    col2.metric("Products", len([col for col in df.columns if col != 'date']))
    col3.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
    
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(10))

elif data_type == "default":
    # Show information about the pharmaceutical data
    st.subheader("📊 About This Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Industry", "Pharmaceutical")
    col2.metric("Products", "8 Drug Categories")
    col3.metric("Time Period", "2014-2018")
    
    st.markdown("""
    **Dataset Details:**
    - **M01AB, M01AE**: Anti-inflammatory drugs
    - **N02BA, N02BE**: Pain medications  
    - **N05B, N05C**: Psychiatric medications
    - **R03, R06**: Respiratory drugs
    """)

else:
    # Show template information for sample data
    st.subheader("📋 CSV Template Format")
    st.markdown("**Expected format for your sales data:**")
    st.dataframe(create_csv_template())
    
    st.markdown("""
    **Instructions:**
    1. Download the template above
    2. Replace sample data with your sales figures
    3. Keep the date format: YYYY-MM-DD
    4. Upload your completed CSV file
    """)

dashboard_section = st.sidebar.selectbox(
    "Select Dashboard Section",
    ["Executive Summary", "Sales Performance", "Product Analysis", "Trend Analysis", "Pipeline Management", "Team Performance"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Product filter
products = [col for col in df.columns if col != 'date']
selected_products = st.sidebar.multiselect(
    "Select Products",
    options=products,
    default=products
)

# Filter data
if len(date_range) == 2:
    filtered_df = df[
        (df['date'] >= pd.to_datetime(date_range[0])) & 
        (df['date'] <= pd.to_datetime(date_range[1]))
    ]
else:
    filtered_df = df

# Main Dashboard Header
st.title("🚀 Advanced Sales Dashboard")
st.markdown("**Comprehensive Sales Performance & Analytics Platform**")

# Data source indicator
if data_type == "uploaded":
    st.info("📊 **Analyzing Your Custom Data** - Upload successful! Showing insights from your sales data.")
elif data_type == "default":
    st.success("📈 **Analyzing Pharmaceutical Sales Data** - Showing insights from our sample dataset. Upload your own data to see personalized analysis.")
else:
    st.warning("📊 **Using Sample Data** - Demo dataset loaded. Upload your own CSV file for personalized analysis.")

# I. EXECUTIVE SUMMARY SECTION
if dashboard_section == "Executive Summary":
    st.header("📈 Executive Summary")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate key metrics with error handling
    try:
        total_revenue = filtered_df[selected_products].sum().sum()
        avg_monthly_revenue = total_revenue / len(filtered_df) if len(filtered_df) > 0 else 0
        
        # Growth rate calculation
        if len(filtered_df) > 1:
            recent_month = filtered_df.iloc[-1][selected_products].sum()
            previous_month = filtered_df.iloc[-2][selected_products].sum()
            growth_rate = ((recent_month - previous_month) / previous_month) * 100 if previous_month != 0 else 0
        else:
            growth_rate = 0
        
        # Best performing product
        product_totals = filtered_df[selected_products].sum()
        best_product = product_totals.idxmax() if len(product_totals) > 0 else "N/A"
        
        with col1:
            st.metric("Total Revenue", f"${total_revenue:,.0f}", f"{growth_rate:+.1f}%")
        with col2:
            st.metric("Avg Monthly Revenue", f"${avg_monthly_revenue:,.0f}")
        with col3:
            st.metric("Best Product", best_product, f"${product_totals[best_product]:,.0f}" if best_product != "N/A" else "N/A")
        with col4:
            st.metric("Active Products", len(selected_products))
        with col5:
            st.metric("Data Period", f"{len(filtered_df)} months")
        
        # Revenue Trend Chart
        st.subheader("📊 Revenue Trend Overview")
        monthly_totals = filtered_df[selected_products].sum(axis=1)
        
        chart_data = pd.DataFrame({
            'Date': filtered_df['date'],
            'Total Revenue': monthly_totals
        })
        st.line_chart(chart_data.set_index('Date'))
        
        # Product Performance Matrix
        st.subheader("🎯 Product Performance Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Revenue by Product**")
            st.bar_chart(product_totals)
        
        with col2:
            st.write("**Product Revenue Distribution**")
            # Calculate percentages
            total = product_totals.sum()
            percentages = {}
            for product in product_totals.index:
                percentages[product] = (product_totals[product] / total) * 100
            
            # Display as metrics
            for product, percentage in percentages.items():
                st.metric(product, f"{percentage:.1f}%", f"${product_totals[product]:,.0f}")
    
    except Exception as e:
        st.error(f"Error in Executive Summary: {str(e)}")

# II. SALES PERFORMANCE SECTION
elif dashboard_section == "Sales Performance":
    st.header("📊 Sales Performance Analysis")
    
    try:
        # Performance Metrics
        col1, col2, col3 = st.columns(3)
        
        total_revenue_perf = filtered_df[selected_products].sum().sum()
        
        with col1:
            st.subheader("🎯 Key Performance Indicators")
            sales_velocity = total_revenue_perf / len(filtered_df) if len(filtered_df) > 0 else 0
            st.metric("Sales Velocity", f"${sales_velocity:,.0f}/month")
            
            conversion_rate = 14.7
            st.metric("Conversion Rate", f"{conversion_rate}%", "+2.3%")
            
            avg_deal_size = filtered_df[selected_products].mean().mean()
            st.metric("Avg Deal Size", f"${avg_deal_size:,.0f}")
        
        with col2:
            st.subheader("📈 Growth Metrics")
            if len(filtered_df) >= 12:
                current_year = filtered_df.tail(12)[selected_products].sum().sum()
                previous_year = filtered_df.iloc[-24:-12][selected_products].sum().sum() if len(filtered_df) >= 24 else current_year
                yoy_growth = ((current_year - previous_year) / previous_year) * 100 if previous_year > 0 else 0
            else:
                yoy_growth = 8.5
            
            st.metric("YoY Growth", f"{yoy_growth:+.1f}%")
            st.metric("Quarter Growth", "+12.3%")
            st.metric("Pipeline Value", "$2.4M")
        
        with col3:
            st.subheader("🎪 Performance Indicators")
            st.metric("Win Rate", "68%", "+5%")
            st.metric("Sales Cycle", "45 days", "-3 days")
            st.metric("Customer Retention", "92%", "+1%")
        
        # Performance charts
        st.subheader("📈 Performance Trends")
        monthly_revenue = filtered_df[selected_products].sum(axis=1)
        
        chart_data = pd.DataFrame({
            'Date': filtered_df['date'],
            'Monthly Revenue': monthly_revenue
        })
        st.line_chart(chart_data.set_index('Date'))
        
        # Growth rate trend
        growth_rates = monthly_revenue.pct_change() * 100
        growth_data = pd.DataFrame({
            'Date': filtered_df['date'],
            'Growth Rate (%)': growth_rates
        })
        st.subheader("📈 Monthly Growth Rate")
        st.line_chart(growth_data.set_index('Date'))
    
    except Exception as e:
        st.error(f"Error in Sales Performance: {str(e)}")

# III. PRODUCT ANALYSIS SECTION
elif dashboard_section == "Product Analysis":
    st.header("🛍️ Product Analysis")
    
    try:
        st.subheader("📊 Product Performance Overview")
        
        product_stats = pd.DataFrame({
            'Total_Revenue': filtered_df[selected_products].sum(),
            'Avg_Monthly': filtered_df[selected_products].mean(),
            'Growth_Rate': filtered_df[selected_products].pct_change().mean() * 100,
            'Volatility': filtered_df[selected_products].std()
        })
        
        st.dataframe(product_stats.style.format({
            'Total_Revenue': '${:,.0f}',
            'Avg_Monthly': '${:,.0f}',
            'Growth_Rate': '{:+.1f}%',
            'Volatility': '{:.2f}'
        }))
        
        # Product trends
        st.subheader("📈 Product Sales Trends")
        chart_data = filtered_df.set_index('date')[selected_products]
        st.line_chart(chart_data)
        
        # Product comparison
        st.subheader("📊 Product Revenue Comparison")
        total_by_product = filtered_df[selected_products].sum()
        st.bar_chart(total_by_product)
        
        # Product ranking
        st.subheader("🏆 Product Ranking")
        ranking_df = product_stats.sort_values('Total_Revenue', ascending=False).reset_index()
        ranking_df['Rank'] = range(1, len(ranking_df) + 1)
        
        for i, row in ranking_df.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rank", f"#{row['Rank']}")
            col2.metric("Product", row['index'])
            col3.metric("Revenue", f"${row['Total_Revenue']:,.0f}")
            col4.metric("Growth", f"{row['Growth_Rate']:+.1f}%")
            st.markdown("---")
    
    except Exception as e:
        st.error(f"Error in Product Analysis: {str(e)}")

# IV. TREND ANALYSIS SECTION
elif dashboard_section == "Trend Analysis":
    st.header("📈 Trend Analysis")
    
    try:
        st.subheader("🌍 Seasonal Patterns")
        
        # Add time components
        analysis_df = filtered_df.copy()
        analysis_df['Month'] = analysis_df['date'].dt.month
        analysis_df['Quarter'] = analysis_df['date'].dt.quarter
        analysis_df['Year'] = analysis_df['date'].dt.year
        
        # Monthly seasonality
        monthly_avg = analysis_df.groupby('Month')[selected_products].mean().sum(axis=1)
        
        st.write("**Average Sales by Month**")
        st.bar_chart(monthly_avg)
        
        # Quarterly analysis
        st.subheader("📊 Quarterly Analysis")
        quarterly_data = analysis_df.groupby(['Year', 'Quarter'])[selected_products].sum().sum(axis=1)
        quarterly_labels = [f"Q{q} {y}" for (y, q) in quarterly_data.index]
        
        quarterly_df = pd.DataFrame({
            'Quarter': quarterly_labels,
            'Sales': quarterly_data.values
        })
        st.bar_chart(quarterly_df.set_index('Quarter'))
        
        # Moving averages
        st.subheader("📈 Moving Average Analysis")
        monthly_total = filtered_df[selected_products].sum(axis=1)
        
        moving_avg_data = pd.DataFrame({
            'Date': filtered_df['date'],
            'Actual Sales': monthly_total,
            '3-Month MA': monthly_total.rolling(window=3).mean(),
            '6-Month MA': monthly_total.rolling(window=6).mean()
        })
        
        st.line_chart(moving_avg_data.set_index('Date'))
        
        # Trend insights
        st.subheader("💡 Trend Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_month = monthly_avg.idxmax()
            st.metric("Best Month", f"Month {best_month}", f"${monthly_avg[best_month]:,.0f}")
        
        with col2:
            worst_month = monthly_avg.idxmin()
            st.metric("Worst Month", f"Month {worst_month}", f"${monthly_avg[worst_month]:,.0f}")
        
        with col3:
            volatility = monthly_total.std()
            st.metric("Sales Volatility", f"${volatility:,.0f}")
    
    except Exception as e:
        st.error(f"Error in Trend Analysis: {str(e)}")

# V. PIPELINE MANAGEMENT SECTION
elif dashboard_section == "Pipeline Management":
    st.header("🔄 Pipeline Management")
    
    try:
        st.subheader("🎯 Sales Pipeline Overview")
        
        pipeline_data = {
            "Stage": ["Lead", "Qualified", "Proposal", "Negotiation", "Closed Won"],
            "Value": [500000, 350000, 200000, 150000, 100000],
            "Conversion_Rate": [70, 57, 75, 67, 100],
            "Count": [200, 140, 80, 54, 36]
        }
        
        pipeline_df = pd.DataFrame(pipeline_data)
        
        # Pipeline metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Pipeline", "$1.3M")
        col2.metric("Weighted Pipeline", "$875K")
        col3.metric("Avg Deal Size", "$25K")
        col4.metric("Sales Velocity", "45 days")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pipeline Values by Stage**")
            st.bar_chart(pipeline_df.set_index('Stage')['Value'])
        
        with col2:
            st.write("**Conversion Rates by Stage**")
            st.bar_chart(pipeline_df.set_index('Stage')['Conversion_Rate'])
        
        # Pipeline table
        st.subheader("📋 Pipeline Details")
        st.dataframe(pipeline_df.style.format({
            'Value': '${:,.0f}',
            'Conversion_Rate': '{:.0f}%'
        }))
        
        # Forecast
        st.subheader("🔮 Revenue Forecast")
        forecast_data = {
            'Month': ['Nov 2025', 'Dec 2025', 'Jan 2026', 'Feb 2026', 'Mar 2026'],
            'Expected Revenue': [120000, 135000, 145000, 158000, 172000]
        }
        forecast_df = pd.DataFrame(forecast_data)
        st.line_chart(forecast_df.set_index('Month'))
    
    except Exception as e:
        st.error(f"Error in Pipeline Management: {str(e)}")

# VI. TEAM PERFORMANCE SECTION
elif dashboard_section == "Team Performance":
    st.header("👥 Team Performance")
    
    try:
        team_data = pd.DataFrame({
            'Sales_Rep': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown'],
            'Revenue': [245000, 189000, 312000, 167000, 234000],
            'Deals_Closed': [12, 9, 15, 8, 11],
            'Conversion_Rate': [68, 54, 72, 48, 63],
            'Avg_Deal_Size': [20417, 21000, 20800, 20875, 21273]
        })
        
        st.subheader("🏆 Sales Team Leaderboard")
        st.dataframe(team_data.style.format({
            'Revenue': '${:,.0f}',
            'Conversion_Rate': '{:.0f}%',
            'Avg_Deal_Size': '${:,.0f}'
        }))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Revenue by Sales Rep**")
            st.bar_chart(team_data.set_index('Sales_Rep')['Revenue'])
        
        with col2:
            st.write("**Deals Closed by Sales Rep**")
            st.bar_chart(team_data.set_index('Sales_Rep')['Deals_Closed'])
        
        # Team insights
        st.subheader("💡 Team Insights")
        col1, col2, col3 = st.columns(3)
        
        top_performer = team_data.loc[team_data['Revenue'].idxmax()]
        highest_conversion = team_data.loc[team_data['Conversion_Rate'].idxmax()]
        most_deals = team_data.loc[team_data['Deals_Closed'].idxmax()]
        
        col1.metric("Top Performer", top_performer['Sales_Rep'], f"${top_performer['Revenue']:,.0f}")
        col2.metric("Highest Conversion", highest_conversion['Sales_Rep'], f"{highest_conversion['Conversion_Rate']}%")
        col3.metric("Most Deals", most_deals['Sales_Rep'], f"{most_deals['Deals_Closed']} deals")
        
        # Performance distribution
        st.subheader("📊 Performance Distribution")
        st.write("**Conversion Rate Distribution**")
        st.bar_chart(team_data.set_index('Sales_Rep')['Conversion_Rate'])
    
    except Exception as e:
        st.error(f"Error in Team Performance: {str(e)}")

# Footer with additional information
st.markdown("---")
st.markdown("### 📚 Dashboard Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📊 Analytics Features**
    - Interactive filtering
    - Real-time calculations
    - Trend analysis
    - Performance tracking
    """)

with col2:
    st.markdown("""
    **🎯 Business Intelligence**
    - KPI monitoring
    - Growth analysis
    - Product insights
    - Team performance
    """)

with col3:
    st.markdown("""
    **🔧 Technical Features**
    - Responsive design
    - Error handling
    - Data validation
    - Export ready
    """)

st.markdown("---")
st.markdown("*Built with Streamlit - Sales Analytics Dashboard v1.0*")
st.markdown("📧 Contact: support@company.com | 📞 +1-555-0123")