"""
PHASE 4: EXOGENOUS MARKET CHARACTERISATION
Fulfills Section 4.5 of the TFM.
Projects the unsupervised categorical cluster time series against physical 
grid variables (Renewables, Demand) and calendar effects.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless mode for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import holidays

# ============================================================================\
# CONFIGURATION - UPDATE THESE WITH YOUR ESIOS FILE DETAILS
# ============================================================================\
OUTPUT_DIR = Path("phase4_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Input files from Phase 3
DEMAND_CSV = r"Phase3_outputs\DEMAND_Cluster_Assignments_k3.csv"
SUPPLY_CSV = r"Phase3_outputs\SUPPLY_Cluster_Assignments_k3.csv"

# Your ESIOS Data File
ESIOS_CSV = "esios_data_2021_2025.csv" 

# Map the column names in your ESIOS file here:
ESIOS_COL_DATE = "Date"
ESIOS_COL_HOUR = "Hour"
ESIOS_COL_WIND = "Wind_MW"
ESIOS_COL_SOLAR = "Solar_PV_MW"
ESIOS_COL_DEMAND = "Demand_Forecast_MW"

# ============================================================================\
# 1. DATA LOADING & MERGING
# ============================================================================\
def load_and_merge_clusters():
    print("Loading Phase 3 Cluster Assignments...")
    df_dem = pd.read_csv(DEMAND_CSV)
    df_sup = pd.read_csv(SUPPLY_CSV)
    
    # Rename for clarity before merging
    df_dem = df_dem.rename(columns={"Cluster_ID": "Demand_Cluster"}).drop(columns=["Side"])
    df_sup = df_sup.rename(columns={"Cluster_ID": "Supply_Cluster"}).drop(columns=["Side"])
    
    # Merge Supply and Demand together
    df_master = pd.merge(df_dem, df_sup, on=["Date", "Hour"], how="inner")
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    
    print(f"Merged Market Data: {len(df_master)} hours.")
    return df_master

def load_or_simulate_esios(dates, hours):
    if Path(ESIOS_CSV).exists():
        print(f"Loading real exogenous data from {ESIOS_CSV}...")
        esios = pd.read_csv(ESIOS_CSV)
        esios[ESIOS_COL_DATE] = pd.to_datetime(esios[ESIOS_COL_DATE])
        return esios
    else:
        print(f"[WARNING] ESIOS file '{ESIOS_CSV}' not found.")
        print("Generating realistic simulated Spanish grid data to test pipeline...")
        np.random.seed(42)
        
        # Simulate realistic solar (peaks in middle of day)
        solar = np.where((hours >= 9) & (hours <= 18), 
                         np.random.normal(15000, 3000, len(hours)), 0)
        solar = np.clip(solar, 0, None)
        
        # Simulate wind (semi-random, higher in winter)
        wind = np.random.normal(8000, 4000, len(hours))
        wind = np.clip(wind, 0, None)
        
        return pd.DataFrame({
            ESIOS_COL_DATE: dates,
            ESIOS_COL_HOUR: hours,
            ESIOS_COL_WIND: wind,
            ESIOS_COL_SOLAR: solar,
            ESIOS_COL_DEMAND: np.random.normal(30000, 5000, len(hours))
        })

# ============================================================================\
# 2. FEATURE ENGINEERING (Calendar Effects)
# ============================================================================\
def add_calendar_features(df):
    print("Engineering Calendar Features (Section 4.5.3)...")
    
    # Day of Week & Weekend Boolean
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df['Is_Weekend'] = df['Date'].dt.dayofweek >= 5
    
    # Spanish Public Holidays
    es_holidays = holidays.Spain(years=df['Date'].dt.year.unique())
    df['Is_Holiday'] = df['Date'].isin(es_holidays)
    
    # Combine Weekend/Holiday into a single "Non-Working Day" flag
    df['Non_Working_Day'] = df['Is_Weekend'] | df['Is_Holiday']
    
    # Seasonality
    month = df['Date'].dt.month
    conditions = [
        (month.isin([12, 1, 2])),
        (month.isin([3, 4, 5])),
        (month.isin([6, 7, 8])),
        (month.isin([9, 10, 11]))
    ]
    choices = ['Winter', 'Spring', 'Summer', 'Autumn']
    df['Season'] = np.select(conditions, choices, default='Unknown')    
    return df

# ============================================================================\
# 3. PLOTTING FUNCTIONS
# ============================================================================\
def plot_market_cross_tabulation(df):
    """How often does Supply Cluster X meet Demand Cluster Y?"""
    print("Plotting Market Intersection...")
    crosstab = pd.crosstab(df['Supply_Cluster'], df['Demand_Cluster'], normalize='all') * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Occurrence Probability (%)'})
    plt.title("Joint Probability of Market Clearing States")
    plt.xlabel("Demand Cluster Archetype")
    plt.ylabel("Supply Cluster Archetype")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Phase4_Market_Intersection_Heatmap.png", dpi=300)
    plt.close()

def plot_vre_penetration(df, target_side="Supply"):
    """Section 4.5.1: Probability of transitioning to clusters based on Solar/Wind"""
    print(f"Plotting VRE Penetration for {target_side}...")
    
    col_cluster = f"{target_side}_Cluster"
    df['Total_VRE'] = df[ESIOS_COL_WIND] + df[ESIOS_COL_SOLAR]
    
    # Create 10 decile bins of Renewable Generation
    df['VRE_Decile'] = pd.qcut(df['Total_VRE'], 10, labels=[f"D{i}" for i in range(1, 11)])
    
    # Calculate conditional probabilities
    vre_dist = pd.crosstab(df['VRE_Decile'], df[col_cluster], normalize='index') * 100
    
    ax = vre_dist.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title(f"{target_side} Archetype Probability given Renewable Penetration")
    plt.xlabel("Renewable Generation Deciles (D1=Low, D10=High)")
    plt.ylabel("Probability (%)")
    plt.legend(title=f"{target_side} Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Phase4_VRE_Impact_{target_side}.png", dpi=300)
    plt.close()

def plot_calendar_decay(df, target_side="Demand"):
    """Section 4.5.3: Structural decay across weekends/holidays"""
    print(f"Plotting Calendar Effects for {target_side}...")
    col_cluster = f"{target_side}_Cluster"
    
    # Calculate probability of clusters on Working vs Non-Working days
    cal_dist = pd.crosstab(df['Non_Working_Day'], df[col_cluster], normalize='index') * 100
    cal_dist.index = ['Working Day', 'Weekend / Holiday']
    
    ax = cal_dist.plot(kind='bar', figsize=(8, 6), colormap='Set2')
    plt.title(f"Calendar Effects: {target_side} Cluster Distribution")
    plt.ylabel("Occurrence Probability (%)")
    plt.xticks(rotation=0)
    plt.legend(title=f"{target_side} Cluster")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"Phase4_Calendar_Effects_{target_side}.png", dpi=300)
    plt.close()

# ============================================================================\
# MAIN EXOGENOUS PIPELINE
# ============================================================================\
if __name__ == "__main__":
    print("\n" + "="*60)
    print("STARTING PHASE 4: EXOGENOUS CHARACTERISATION")
    print("="*60)
    
    # 1. Load Data
    df_market = load_and_merge_clusters()
    df_esios = load_or_simulate_esios(df_market['Date'], df_market['Hour'])
    
    # 2. Merge all together
    df = pd.merge(df_market, df_esios, on=[ESIOS_COL_DATE, ESIOS_COL_HOUR], how="left")
    
    # 3. Add Thesis specific features
    df = add_calendar_features(df)
    
    # Save the master dataset for your thesis archive
    df.to_csv(OUTPUT_DIR / "Phase4_Master_Exogenous_Dataset.csv", index=False)
    
    # 4. Generate Thesis Visuals
    plot_market_cross_tabulation(df)
    
    # Run profiles for both sides
    for side in ["Supply", "Demand"]:
        plot_vre_penetration(df, target_side=side)
        plot_calendar_decay(df, target_side=side)
        
    print("\n[DONE] PHASE 4 COMPLETE! Check the 'phase4_output' directory for your thesis plots.")