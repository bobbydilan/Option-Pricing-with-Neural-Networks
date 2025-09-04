"""
Data Explorer

Given the path to your project root, this script scans the `00Data/` folder for CSV files and prints a quick structural
summary for each file:
    - file name & relative path
    - number of rows / columns
    - column names & dtypes
    - first 3 rows
    - auto-detected date-like columns
    - enhanced analysis for economic/uncertainty data
"""
#%%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

#%%
# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate from 01General up to THESIS folder, then into 00Data
PROJECT_ROOT = Path(BASE_DIR).parent  # Go up: 01General -> Code -> THESIS
DATA_PATH = PROJECT_ROOT / "00Data"

#%%
def _summarise_csv(csv_path: Path, sample_rows: int = None) -> None:
    """Print a concise summary of a CSV file."""
    rel_path = csv_path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in csv_path.parents else csv_path.name
    print(f"\n{rel_path}")
    print("-" * 50)

    try:
        df = pd.read_csv(csv_path, nrows=sample_rows)
    except Exception as exc:
        print(f"Could not read file: {exc}")
        return

    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("dtypes:\n", df.dtypes)
    print("\nFirst 3 rows:")
    print(df.head(3))

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "time", "day"))]
    if date_cols:
        print("Date-like columns:", date_cols)


def _analyze_economic_data(df: pd.DataFrame, file_name: str):
    """Enhanced analysis for economic/uncertainty data files."""
    # Check if this is one of the economic/uncertainty files
    economic_files = {
        'Economic Policy Uncertainty Index for United States.csv': 'USEPUINDXD',
        'Equity Market Economic Uncertainty Index.csv': 'WLEMUINDXD', 
        'Equity Market Volatility.csv': 'INFECTDISEMVTRACKD'
    }
    
    if file_name not in economic_files:
        return
        
    value_col = economic_files[file_name]
    print(f"\n--- Enhanced Analysis for {file_name} ---")
    
    # Basic statistics
    if value_col in df.columns:
        print(f"\n{value_col} Statistics:")
        print(f"  Mean: {df[value_col].mean():.2f}")
        print(f"  Std:  {df[value_col].std():.2f}")
        print(f"  Min:  {df[value_col].min():.2f}")
        print(f"  Max:  {df[value_col].max():.2f}")
        print(f"  Missing values: {df[value_col].isna().sum()}")
        
        # Date range
        if 'observation_date' in df.columns:
            df_temp = df.copy()
            df_temp['observation_date'] = pd.to_datetime(df_temp['observation_date'])
            print(f"  Date range: {df_temp['observation_date'].min().date()} to {df_temp['observation_date'].max().date()}")
            print(f"  Total observations: {len(df_temp)}")

def _analyze_final_dataset(df: pd.DataFrame):
    """Enhanced analysis for the final options dataset with new columns."""
    print("\n--- Enhanced Analysis for Final Options Dataset ---")
    
    # Check for new economic/uncertainty columns
    new_cols = ['epu_index', 'equity_uncertainty', 'equity_volatility', 'open_interest']
    present_cols = [col for col in new_cols if col in df.columns]
    
    if present_cols:
        print(f"\nNew Economic/Uncertainty Columns Found: {present_cols}")
        
        for col in present_cols:
            if col in df.columns:
                print(f"\n{col} Statistics:")
                print(f"  Mean: {df[col].mean():.2f}")
                print(f"  Std:  {df[col].std():.2f}")
                print(f"  Min:  {df[col].min():.2f}")
                print(f"  Max:  {df[col].max():.2f}")
                print(f"  Missing values: {df[col].isna().sum()} ({df[col].isna().mean()*100:.1f}%)")
        
        if 'mid_price' in df.columns and present_cols:
            print(f"\nCorrelations with Option Mid Price:")
            for col in present_cols:
                if col in df.columns and not df[col].isna().all():
                    corr = df['mid_price'].corr(df[col])
                    print(f"  {col}: {corr:.3f}")

def explore_data(data_directory: Path = None):
    """Explore CSV files in the specified data directory."""
    if data_directory is None:
        data_directory = DATA_PATH
    
    data_dir = Path(data_directory)
    
    if not data_dir.exists():
        print(f"Data directory does not exist: {data_dir}")
        print(f"Looking for data in: {data_dir.absolute()}")
        return
    
    if not data_dir.is_dir():
        print(f"Path is not a directory: {data_dir}")
        return

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        print(f"Directory contents: {list(data_dir.iterdir())}")
        return

    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}:")
    for csv_file in csv_files:
        _summarise_csv(csv_file)
        
        try:
            df = pd.read_csv(csv_file, nrows=10000)  # Sample for analysis
            _analyze_economic_data(df, csv_file.name)
            
            if 'final' in csv_file.name.lower() and len(df.columns) > 20:
                _analyze_final_dataset(df)
        except Exception as e:
            print(f"Could not perform enhanced analysis: {e}")

if __name__ == "__main__":
    print(f"Script location: {Path(__file__).absolute()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {DATA_PATH}")
    print(f"Data path exists: {DATA_PATH.exists()}")
    explore_data()