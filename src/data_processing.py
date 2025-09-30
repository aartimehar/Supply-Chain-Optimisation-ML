"""
data_processing.py
Functions to load, clean, and merge datasets for pharma supply chain project.
"""

import pandas as pd

def load_sales_data(filepath="../Data/salesweekly.csv"):
    """Load historical pharmaceutical weekly sales data"""
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df

def load_fda_shortages(filepath="../Data/fda_shortages.csv"):
    """Load FDA drug shortage data"""
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    df['shortage_start'] = pd.to_datetime(df['shortage_start'])
    df['shortage_end'] = pd.to_datetime(df['shortage_end'], errors='coerce')
    return df

def load_synthetic_biomanufacturing(filepath="../Data/synthetic_biomanufacturing.csv"):
    """Load synthetic biomanufacturing dataset"""
    df = pd.read_csv(filepath)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    return df

def merge_datasets(sales_df, shortages_df, bio_df):
    """
    Merge all datasets into a single DataFrame for exploration.
    """
    # Merge sales with shortages
    df = sales_df.merge(shortages_df, how='left', left_on='drug_name', right_on='drug_name')
    
    # Merge with synthetic biomanufacturing data on closest date (weekly)
    df = df.merge(bio_df, how='left', on='date')
    
    # Fill missing shortage status
    df['status'] = df['status'].fillna("no_shortage")
    
    return df
