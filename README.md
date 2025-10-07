# ATLAS Sales Intelligence Platform

🚀 **[Live Demo](https://aartimehar-supply-ch-dashboardstreamlit-native-dashboard-lf6aqg.streamlit.app/)**

A modern web application for sales data analysis and business intelligence. Built with Streamlit, this platform provides comprehensive analytics dashboards for sales teams and business analysts.

## Use Case

This platform solves the common challenge of scattered sales data and manual reporting processes. Sales teams often struggle with:

- **Data Silos**: Sales data trapped in spreadsheets, CRM systems, and different departments
- **Manual Reporting**: Hours spent creating weekly/monthly reports instead of analyzing trends
- **Limited Insights**: Basic charts that don't reveal actionable business intelligence
- **Accessibility**: Technical barriers preventing non-technical users from exploring data

ATLAS bridges this gap by providing an intuitive interface where users can upload their CSV sales data and immediately access professional-grade analytics. Whether you're a company tracking ATC categories, a retail business monitoring product performance, or a SaaS company analyzing subscription revenue, the platform adapts to your data structure and provides relevant insights.

The dual-mode design lets teams explore demo data to understand capabilities, then seamlessly transition to analyzing their own business data using the same powerful analytics suite.

## Features

- **Dual Data Sources**: Analyze demo data or upload custom CSV files
- **Real-time Analytics**: Interactive dashboards with filtering and drill-down capabilities  
- **CSV Template System**: Guided data upload with validation and error handling
- **Multi-currency Support**: Display data in USD, EUR, or GBP
- **Advanced Filtering**: Date ranges, product selection, and revenue thresholds
- **Executive Reporting**: Six specialized analytics modules for different business needs

## Interesting Techniques

The codebase demonstrates several modern web development patterns:

- **[FileReader API](https://developer.mozilla.org/en-US/docs/Web/API/FileReader)** integration via Streamlit's file uploader for client-side CSV processing
- **Pandas vectorized operations** for efficient data transformation and aggregation
- **State management patterns** using Streamlit's session state for maintaining user preferences
- **Data validation pipelines** with comprehensive error handling and user feedback
- **Responsive layout design** using Streamlit's column system and container width controls
- **Real-time chart updates** with native Streamlit charting components

## Technology Stack

**Core Framework:**
- [Streamlit](https://streamlit.io/) - Web application framework for data science
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NumPy](https://numpy.org/) - Numerical computing

**Data Processing:**
- CSV parsing with encoding detection
- Date normalization and timezone handling  
- Revenue aggregation and growth rate calculations

**UI Components:**
- Native Streamlit charts (line, bar, area)
- Multi-select widgets with dynamic options
- File upload with progress indicators
- Expandable sections for data preview

## Project Structure

```
├── dashboard/
├── Data/
├── notebooks/
├── reports/
├── src/
├── requirements.txt
└── README.md
```

**Key Directories:**
- [`dashboard/`](./dashboard/) - Main application files including the Streamlit dashboard
- [`Data/`](./Data/) - Sample datasets including sales data and FDA shortage information
- [`notebooks/`](./notebooks/) - Jupyter notebooks for data exploration and analysis
- [`src/`](./src/) - Core Python modules for data processing and modeling
- [`reports/`](./reports/) - Generated insights and findings documentation

## How to Use

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd supply-chain-optimisation-ml
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install streamlit pandas numpy
```

### Running the Application

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Launch the Streamlit application:
```bash
streamlit run streamlit_native_dashboard.py
```

3. Open your browser to `http://localhost:8501`

### Using Your Own Data

1. **Download Template**: Click "Download CSV Template" in the sidebar
2. **Format Your Data**: Fill the template with your sales data
3. **Upload File**: Use the file uploader to import your CSV
4. **Analyze**: Access all analytics modules with your data

The platform accepts CSV files with date columns and numeric sales data. The system automatically detects and validates your data structure.
