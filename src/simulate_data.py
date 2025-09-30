"""
Simulate synthetic biomanufacturing data for pharma supply chain project.
Generates weekly data for raw materials, batch yield, and bioreactor utilization.
"""

import pandas as pd
import numpy as np

def generate_synthetic_biomanufacturing_data(n_weeks=52, start_date="2023-01-01"):
    np.random.seed(42)  # reproducible results

    # Weekly dates
    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W")

    # Example raw materials
    raw_materials = ["MediaA", "MediaB", "ReagentX", "ReagentY"]

    data = []

    for date in dates:
        for rm in raw_materials:
            quantity_used = np.random.randint(400, 600)
            batch_yield = int(quantity_used * np.random.uniform(0.85, 0.95))
            utilization = round(np.random.uniform(0.7, 0.95), 2)

            data.append({
                "date": date,
                "raw_material": rm,
                "quantity_used": quantity_used,
                "batch_yield": batch_yield,
                "bioreactor_utilization": utilization
            })

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    df = generate_synthetic_biomanufacturing_data(n_weeks=60, start_date="2023-01-01")
    output_path = "../data/synthetic_biomanufacturing.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Synthetic biomanufacturing data saved to {output_path}")
