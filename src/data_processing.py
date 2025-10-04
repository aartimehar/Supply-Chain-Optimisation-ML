import pandas as pd

# =====================================================
# 🧪 Data Loading Functions
# =====================================================

def load_sales_data(path="../data/salesweekly.csv"):
    df = pd.read_csv(path)

    # Strip spaces and lowercase all column names
    df.columns = df.columns.str.strip().str.lower()

    # Rename 'datum' to 'date'
    if "datum" in df.columns:
        df = df.rename(columns={"datum": "date"})
    elif "date" not in df.columns:
        raise KeyError("❌ Could not find a 'date' or 'datum' column in the sales file.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Melt from wide → long
    df = df.melt(
        id_vars="date",
        var_name="product_code",
        value_name="units_sold"
    )

    df = df.dropna(subset=["units_sold"])
    df = df[df["units_sold"] > 0]

    return df




def load_fda_shortages(path="../data/fda_shortages.csv"):
    """
    Load FDA drug shortages dataset.
    Columns could include: ['generic_name', 'status', 'date_updated']
    """
    df = pd.read_csv(path)
    if "date_updated" in df.columns:
        df["date_updated"] = pd.to_datetime(df["date_updated"], errors="coerce")
    return df


def load_synthetic_biomanufacturing(path="../data/synthetic_biomanufacturing.csv"):
    """
    Load synthetic biomanufacturing data.
    Columns: ['date', 'raw_material', 'batch_yield', 'production_cost']
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# =====================================================
# 🔗 Merge Function
# =====================================================

def merge_datasets(sales_df, shortages_df, bio_df):
    """
    Merge sales, shortage, and biomanufacturing data by nearest date.
    """
    # Merge sales with biomanufacturing (on date)
    merged = pd.merge_asof(
        sales_df.sort_values("date"),
        bio_df.sort_values("date"),
        on="date",
        direction="nearest"
    )

    # Add a column showing shortage presence if available
    if "generic_name" in shortages_df.columns:
        shortages_df = shortages_df.rename(columns={"date_updated": "shortage_date"})
        merged["in_shortage"] = merged["product_code"].isin(shortages_df["generic_name"]).astype(int)
    else:
        merged["in_shortage"] = 0

    return merged
