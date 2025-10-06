import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ATLAS Sales Intelligence Platform", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# Template function
def create_csv_template():
    template_data = {
        'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'product_name': ['Product A', 'Product B', 'Product C', 'Product A', 'Product B'],
        'category': ['Category 1', 'Category 2', 'Category 1', 'Category 1', 'Category 2'],
        'revenue': [1250.00, 890.50, 1100.75, 1350.25, 975.00],
        'sales_rep': ['John Smith', 'Jane Doe', 'Mike Johnson', 'John Smith', 'Jane Doe'],
        'quantity': [5, 3, 4, 6, 3]
    }
    return pd.DataFrame(template_data)

# Data loading function
@st.cache_data
def load_data():
    try:
        possible_paths = [
            'Data/salesmonthly.csv',
            '../Data/salesmonthly.csv',
            'C:/Users/aarti/Documents/Supply Chain Optimisation ML/Data/salesmonthly.csv'
        ]
        
        df = None
        for path in possible_paths:
            try:
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
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            dates = pd.date_range('2020-01-01', periods=48, freq='M')
            dummy_data = {
                'date': dates,
                'product_a': np.random.randint(500, 2000, 48),
                'product_b': np.random.randint(300, 1500, 48),
                'product_c': np.random.randint(400, 1800, 48),
                'product_d': np.random.randint(200, 1200, 48)
            }
            df = pd.DataFrame(dummy_data)
        else:
            df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '').str.lower()
            if 'datum' in df.columns:
                df = df.rename(columns={'datum': 'date'})
            df['date'] = pd.to_datetime(df['date'])
            numeric_cols = [col for col in df.columns if col != 'date']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Main dashboard
st.title("⚡ ATLAS Sales Intelligence Platform")
st.markdown("**Professional Analytics • Universal Application • Instant Insights**")

# Load data
df = load_data()

if df is not None:
    # Sidebar
    st.sidebar.title("⚡ ATLAS Analytics")
    st.sidebar.markdown("*Carry Your Business Data with Confidence*")
    
    # Template download
    template_df = create_csv_template()
    csv_template = template_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Download CSV Template",
        data=csv_template,
        file_name="sales_data_template.csv",
        mime="text/csv"
    )
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Analytics Modules")
    dashboard_sections = [
        "Executive Analytics Suite",
        "Revenue Intelligence", 
        "Product Performance Analytics",
        "Trend Analysis & Forecasting",
        "Team Performance Insights",
        "Data Quality & Metrics"
    ]
    
    dashboard_section = st.sidebar.selectbox("Choose Analytics Module:", dashboard_sections)
    
    # Data Filters Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Data Filters")
    
    # Date Range Filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "📅 Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Filter data by date range"
    )
    
    # Product Selection Filter
    product_cols = [col for col in df.columns if col != 'date']
    selected_products = st.sidebar.multiselect(
        "🏷️ Select Products",
        options=product_cols,
        default=product_cols,
        help="Choose which products to include in analysis"
    )
    
    # Revenue Range Filter
    if len(selected_products) > 0:
        total_revenue_range = df[selected_products].sum(axis=1)
        min_revenue = int(total_revenue_range.min())
        max_revenue = int(total_revenue_range.max())
        
        revenue_filter = st.sidebar.slider(
            "💰 Revenue Range Filter",
            min_value=min_revenue,
            max_value=max_revenue,
            value=(min_revenue, max_revenue),
            step=100,
            format="$%d",
            help="Filter periods by total revenue"
        )
    else:
        revenue_filter = (0, 999999999)
    
    # Time Grouping Options
    time_grouping = st.sidebar.selectbox(
        "📈 Time Grouping",
        options=["Monthly", "Quarterly", "Yearly"],
        index=0,
        help="How to group time-series data"
    )
    
    # Top N Products Filter
    top_n = st.sidebar.slider(
        "🔝 Show Top N Products",
        min_value=1,
        max_value=len(product_cols),
        value=min(5, len(product_cols)),
        help="Number of top products to display in rankings"
    )
    
    # Display Options
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Display Options")
    
    show_percentages = st.sidebar.checkbox(
        "📊 Show Percentages",
        value=True,
        help="Display percentage values alongside absolute numbers"
    )
    
    show_growth_indicators = st.sidebar.checkbox(
        "📈 Show Growth Indicators",
        value=True,
        help="Display growth arrows and trend indicators"
    )
    
    currency_format = st.sidebar.selectbox(
        "💱 Currency Format",
        options=["USD ($)", "EUR (€)", "GBP (£)", "Numbers Only"],
        index=0,
        help="Choose currency display format"
    )
    
    # Apply Filters to Data
    filtered_df = df.copy()
    
    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= start_date) & 
            (filtered_df['date'].dt.date <= end_date)
        ]
    
    # Apply product filter
    if selected_products:
        filtered_cols = ['date'] + selected_products
        filtered_df = filtered_df[filtered_cols]
    else:
        st.warning("⚠️ Please select at least one product to display data.")
        st.stop()
    
    # Apply revenue filter
    if len(selected_products) > 0:
        period_revenues = filtered_df[selected_products].sum(axis=1)
        revenue_mask = (period_revenues >= revenue_filter[0]) & (period_revenues <= revenue_filter[1])
        filtered_df = filtered_df[revenue_mask]
    
    # Show filter summary
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Filter Summary")
    st.sidebar.write(f"**📅 Date Range:** {len(filtered_df)} periods")
    st.sidebar.write(f"**🏷️ Products:** {len(selected_products)} selected")
    st.sidebar.write(f"**💰 Revenue Range:** ${revenue_filter[0]:,} - ${revenue_filter[1]:,}")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
        st.stop()
    
    # Calculate key metrics using filtered data
    filtered_product_cols = [col for col in filtered_df.columns if col != 'date']
    total_revenue = filtered_df[filtered_product_cols].sum().sum()
    avg_monthly_revenue = filtered_df[filtered_product_cols].sum(axis=1).mean()
    
    if len(filtered_product_cols) > 0:
        top_product = filtered_df[filtered_product_cols].sum().idxmax()
        top_product_revenue = filtered_df[filtered_product_cols].sum().max()
    else:
        top_product = "N/A"
        top_product_revenue = 0
    
    # Helper function for currency formatting
    def format_currency(amount):
        if currency_format == "USD ($)":
            return f"${amount:,.0f}"
        elif currency_format == "EUR (€)":
            return f"€{amount:,.0f}"
        elif currency_format == "GBP (£)":
            return f"£{amount:,.0f}"
        else:
            return f"{amount:,.0f}"
    
    # Helper function for grouping data by time
    def group_by_time(df, cols, grouping):
        df_grouped = df.set_index('date')
        if grouping == "Quarterly":
            return df_grouped[cols].resample('Q').sum()
        elif grouping == "Yearly":
            return df_grouped[cols].resample('Y').sum()
        else:  # Monthly
            return df_grouped[cols].resample('M').sum()
    
    # Dashboard sections
    if dashboard_section == "Executive Analytics Suite":
        st.header("⚡ Executive Analytics Suite")
        st.markdown(f"*Analyzing {len(filtered_df)} periods • {len(filtered_product_cols)} products • {time_grouping} view*")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate growth for indicators
        if len(filtered_df) > 1 and show_growth_indicators:
            recent_revenue = filtered_df[filtered_product_cols].sum(axis=1).iloc[-1]
            previous_revenue = filtered_df[filtered_product_cols].sum(axis=1).iloc[-2]
            revenue_change = ((recent_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
            revenue_delta = f"{revenue_change:+.1f}%" if show_growth_indicators else None
        else:
            revenue_delta = None
        
        col1.metric("Total Revenue", format_currency(total_revenue), delta=revenue_delta)
        col2.metric("Avg Period", format_currency(avg_monthly_revenue))
        col3.metric("Top Product", top_product.replace('_', ' ').title() if top_product != "N/A" else "N/A")
        col4.metric("Top Revenue", format_currency(top_product_revenue))
        
        # Grouped revenue trend
        st.markdown(f"### 📈 Revenue Trend ({time_grouping})")
        grouped_data = group_by_time(filtered_df, filtered_product_cols, time_grouping)
        revenue_trend = grouped_data.sum(axis=1)
        st.line_chart(revenue_trend)
        
        # Top N products performance
        st.markdown(f"### 🏆 Top {top_n} Product Performance")
        product_totals = filtered_df[filtered_product_cols].sum().sort_values(ascending=False).head(top_n)
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(product_totals)
        
        with col2:
            if show_percentages:
                total_sum = product_totals.sum()
                percentages = (product_totals / total_sum * 100).round(1)
                st.markdown("**Market Share:**")
                for product, pct in percentages.items():
                    st.write(f"• {product.replace('_', ' ').title()}: {pct}%")
    
    elif dashboard_section == "Revenue Intelligence":
        st.header("📈 Revenue Intelligence")
        st.markdown(f"*Revenue velocity tracking • Growth analysis • {time_grouping} performance*")
        
        # Group revenue data by selected time period
        grouped_revenue = group_by_time(filtered_df, filtered_product_cols, time_grouping)
        df_revenue = grouped_revenue.sum(axis=1)
        
        col1, col2, col3, col4 = st.columns(4)
        
        growth_rate = ((df_revenue.iloc[-1] - df_revenue.iloc[0]) / df_revenue.iloc[0] * 100) if len(df_revenue) > 1 else 0
        growth_delta = f"{growth_rate:+.1f}%" if show_growth_indicators else None
        
        col1.metric("Total Revenue", format_currency(df_revenue.sum()))
        col2.metric(f"Avg {time_grouping[:-2]}", format_currency(df_revenue.mean()))
        col3.metric("Growth Rate", f"{growth_rate:.1f}%", delta=growth_delta)
        col4.metric(f"Peak {time_grouping[:-2]}", format_currency(df_revenue.max()))
        
        # Revenue analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📊 Revenue Trend ({time_grouping})")
            st.line_chart(df_revenue)
        
        with col2:
            st.markdown("### 📈 Growth Rate Analysis")
            if len(df_revenue) > 1:
                growth_data = df_revenue.pct_change() * 100
                st.line_chart(growth_data.fillna(0))
            else:
                st.info("Need more data points for growth analysis")
    
    elif dashboard_section == "Product Performance Analytics":
        st.header("🎯 Product Performance Analytics")
        st.markdown(f"*Product rankings • Performance comparison • Top {top_n} analysis*")
        
        product_performance = filtered_df[filtered_product_cols].sum().sort_values(ascending=False)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Top Product", product_performance.index[0].replace('_', ' ').title() if len(product_performance) > 0 else "N/A")
        col2.metric("Top Revenue", format_currency(product_performance.iloc[0]) if len(product_performance) > 0 else format_currency(0))
        col3.metric("Total Products", len(filtered_product_cols))
        col4.metric("Avg per Product", format_currency(product_performance.mean()) if len(product_performance) > 0 else format_currency(0))
        
        # Product rankings and analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🏆 Top {top_n} Product Rankings")
            top_products = product_performance.head(top_n)
            st.bar_chart(top_products)
            
            if show_percentages:
                st.markdown("**Performance Share:**")
                total_performance = product_performance.sum()
                for product, value in top_products.items():
                    pct = (value / total_performance * 100) if total_performance > 0 else 0
                    st.write(f"• {product.replace('_', ' ').title()}: {pct:.1f}%")
        
        with col2:
            st.markdown(f"### 📈 Top {min(3, top_n)} Product Trends ({time_grouping})")
            if len(product_performance) > 0:
                top_products_for_trend = product_performance.head(min(3, top_n)).index
                grouped_trend_data = group_by_time(filtered_df, list(top_products_for_trend), time_grouping)
                st.line_chart(grouped_trend_data)
            else:
                st.info("No product data available")
    
    elif dashboard_section == "Trend Analysis & Forecasting":
        st.header("📈 Trend Analysis & Forecasting")
        st.markdown(f"*Pattern recognition • {time_grouping} analysis • Seasonal insights*")
        
        # Group data by selected time period
        grouped_data = group_by_time(filtered_df, filtered_product_cols, time_grouping)
        df_trend = grouped_data.sum(axis=1)
        
        if len(df_trend) > 1:
            recent_periods = max(1, len(df_trend) // 4)  # Last 25% of periods
            recent_avg = df_trend.tail(recent_periods).mean()
            earlier_avg = df_trend.head(recent_periods).mean()
            trend_direction = "📈 Upward" if recent_avg > earlier_avg else "📉 Downward"
            trend_strength = abs((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0
        else:
            recent_avg = df_trend.iloc[0] if len(df_trend) > 0 else 0
            earlier_avg = recent_avg
            trend_direction = "⟡️ Stable"
            trend_strength = 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Trend Direction", trend_direction)
        col2.metric("Trend Strength", f"{trend_strength:.1f}%")
        col3.metric(f"Recent Avg", format_currency(recent_avg))
        col4.metric(f"Earlier Avg", format_currency(earlier_avg))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 📈 Historical Trend ({time_grouping})")
            st.line_chart(df_trend)
        
        with col2:
            if len(df_trend) > 4:
                st.markdown("### 🔄 Period-over-Period Growth")
                growth_rates = df_trend.pct_change() * 100
                st.line_chart(growth_rates.fillna(0))
            else:
                st.markdown("### 📊 Trend Summary")
                st.write(f"• **Data Points:** {len(df_trend)}")
                st.write(f"• **Highest:** {format_currency(df_trend.max())}")
                st.write(f"• **Lowest:** {format_currency(df_trend.min())}")
                st.write(f"• **Average:** {format_currency(df_trend.mean())}")
    
    elif dashboard_section == "Team Performance Insights":
        st.header("👥 Team Performance Insights")
        st.info("💡 **ATLAS Intelligence:** Simulating team performance based on filtered product data patterns")
        
        team_members = ['Sarah Johnson', 'Mike Chen', 'Emma Rodriguez', 'David Kim', 'Lisa Thompson'][:len(filtered_product_cols)]
        team_performance = {}
        for i, member in enumerate(team_members):
            if i < len(filtered_product_cols):
                team_performance[member] = filtered_df[filtered_product_cols[i]].sum()
        
        if team_performance:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Team Size", len(team_members))
            col2.metric("Top Performer", max(team_performance, key=team_performance.get))
            col3.metric("Top Performance", format_currency(max(team_performance.values())))
            col4.metric("Team Average", format_currency(np.mean(list(team_performance.values()))))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Team Rankings")
                team_df = pd.Series(team_performance).sort_values(ascending=False)
                st.bar_chart(team_df)
            
            with col2:
                if show_percentages:
                    st.markdown("### 📊 Performance Distribution")
                    total_team_performance = sum(team_performance.values())
                    for member, performance in team_df.items():
                        pct = (performance / total_team_performance * 100) if total_team_performance > 0 else 0
                        st.write(f"• {member}: {pct:.1f}%")
        else:
            st.warning("No team data available with current filters")
    
    elif dashboard_section == "Data Quality & Metrics":
        st.header("📋 Data Quality & Metrics")
        st.markdown(f"*Data assessment • Quality indicators • Export options*")
        
        total_records = len(filtered_df)
        complete_records = filtered_df.dropna().shape[0]
        completeness_rate = (complete_records / total_records) * 100 if total_records > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filtered Records", total_records)
        col2.metric("Complete Records", complete_records)
        col3.metric("Data Quality", f"{completeness_rate:.1f}%")
        col4.metric("Selected Products", len(filtered_product_cols))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Data Preview")
            st.dataframe(filtered_df.head(10))
            
            # Data completeness by column
            if len(filtered_df) > 0:
                st.markdown("### 📏 Column Completeness")
                completeness = {}
                for col in filtered_df.columns:
                    non_null_count = filtered_df[col].notna().sum()
                    completeness[col] = (non_null_count / len(filtered_df)) * 100
                completeness_df = pd.Series(completeness)
                st.bar_chart(completeness_df)
        
        with col2:
            st.markdown("### 📄 Export Options")
            
            # Export filtered data
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_data,
                file_name=f"atlas_filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download the currently filtered dataset"
            )
            
            # Export summary report
            summary_data = {
                'Metric': ['Total Records', 'Date Range', 'Total Revenue', 'Avg Revenue', 'Top Product'],
                'Value': [
                    len(filtered_df),
                    f"{filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}",
                    format_currency(total_revenue),
                    format_currency(avg_monthly_revenue),
                    top_product.replace('_', ' ').title() if top_product != "N/A" else "N/A"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_csv = summary_df.to_csv(index=False)
            
            st.download_button(
                label="📈 Download Summary Report",
                data=summary_csv,
                file_name=f"atlas_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download a summary report of key metrics"
            )
            
            # Filter settings info
            st.markdown("### ⚙️ Current Filters")
            st.write(f"• **Date Range:** {date_range[0]} to {date_range[1]}" if len(date_range) == 2 else "All dates")
            st.write(f"• **Products:** {', '.join([p.replace('_', ' ').title() for p in selected_products[:3]])}{' ...' if len(selected_products) > 3 else ''}")
            st.write(f"• **Revenue Filter:** {format_currency(revenue_filter[0])} - {format_currency(revenue_filter[1])}")
            st.write(f"• **Time Grouping:** {time_grouping}")

st.markdown("---")
st.markdown("*© 2024 ATLAS Analytics. Professional Intelligence for Every Business.*")
