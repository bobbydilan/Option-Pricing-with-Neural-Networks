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
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

#%%
# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate from 01General up to THESIS folder, then into 00Data

PROJECT_ROOT = Path(BASE_DIR).parent
DATA_PATH = PROJECT_ROOT / "00Data"

# Filtering configuration (matching MLP1 settings)
APPLY_FILTERS = False  # Set to True to enable filtering
ZERO_VOLUME_INCLUSION = 0.5
FILTER_ITM_OPTIONS = True
MIN_MONEYNESS = 0.5
MAX_MONEYNESS = 1.5
MONEYNESS_LOWER_BOUND = MIN_MONEYNESS
MONEYNESS_UPPER_BOUND = MAX_MONEYNESS
FILTER_SHORT_TERM = True
MIN_DAYS_TO_MATURITY = 0
MAX_DAYS_TO_MATURITY = 750
FILTER_VALID_SPREAD = False

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

    # Moneyness (on raw if available via spx_close & strike_price)
    if {'spx_close', 'strike_price'}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            mny = (df['spx_close'] / df['strike_price']).replace([np.inf, -np.inf], np.nan)
        mny = mny.dropna()
        if len(mny) > 0:
            print("MONEYNESS (S/K) STATISTICS")
            print("-" * 40)
            print(f"Range: {mny.min():.4f} - {mny.max():.4f}")
            print(f"Mean: {mny.mean():.4f}")
            print(f"Median: {mny.median():.4f}")
            # Percentiles
            for p in [1,5,25,50,75,95,99]:
                print(f"P{p}: {mny.quantile(p/100):.4f}")
            # Buckets
            bins = [0, 0.9, 1.1, np.inf]
            labels = ['OTM', 'ATM', 'ITM']
            cats = pd.cut(mny, bins=bins, labels=labels)
            print("Bucket Distribution:")
            for lab in labels:
                cnt = (cats == lab).sum()
                print(f"  {lab}: {cnt:,} ({cnt/len(mny)*100:.1f}%)")
            print()

def apply_data_filters(df):
    """Apply data quality filters matching MLP1 settings."""
    if not APPLY_FILTERS:
        return df
    
    print(f"[FILTER] Starting with {len(df):,} options")
    original_count = len(df)
    
    # Zero volume filter
    if ZERO_VOLUME_INCLUSION < 1.0 and 'volume' in df.columns:
        zero_vol_mask = df['volume'] == 0
        zero_vol_count = zero_vol_mask.sum()
        keep_zero_vol = int(zero_vol_count * ZERO_VOLUME_INCLUSION)
        
        # Keep all non-zero volume + random sample of zero volume
        non_zero_mask = ~zero_vol_mask
        zero_vol_indices = df[zero_vol_mask].sample(n=keep_zero_vol, random_state=42).index
        df = df[non_zero_mask | df.index.isin(zero_vol_indices)]
        print(f"[FILTER] Zero volume: kept {keep_zero_vol:,} of {zero_vol_count:,} zero-volume options")
    
    # ITM options filter
    if FILTER_ITM_OPTIONS and 'moneyness' in df.columns:
        before = len(df)
        df = df[(df['moneyness'] >= MONEYNESS_LOWER_BOUND) & (df['moneyness'] <= MONEYNESS_UPPER_BOUND)]
        print(f"[FILTER] Moneyness [{MONEYNESS_LOWER_BOUND}, {MONEYNESS_UPPER_BOUND}]: {before:,} -> {len(df):,}")
    
    # Short-term filter
    if FILTER_SHORT_TERM and 'days_to_maturity' in df.columns:
        before = len(df)
        df = df[(df['days_to_maturity'] >= MIN_DAYS_TO_MATURITY) & (df['days_to_maturity'] <= MAX_DAYS_TO_MATURITY)]
        print(f"[FILTER] Days to maturity [{MIN_DAYS_TO_MATURITY}, {MAX_DAYS_TO_MATURITY}]: {before:,} -> {len(df):,}")
    
    # Valid spread filter
    if FILTER_VALID_SPREAD and 'best_bid' in df.columns and 'best_offer' in df.columns:
        before = len(df)
        spread = df['best_offer'] - df['best_bid']
        df = df[spread >= 0]
        print(f"[FILTER] Valid spreads: {before:,} -> {len(df):,}")
    
    print(f"[FILTER] Final dataset: {len(df):,} options ({len(df)/original_count*100:.1f}% of original)")
    return df

def generate_summary_statistics_for_final_dataset():
    """Compute and export summary statistics for each numeric column in the final dataset.

    The function creates an exportable CSV with the following columns for every numeric feature:
    Mean, SD, Min, p25, Median, p75, Max, Skewness, Kurtosis.

    It also appends a human-readable table to `eda_all_figures/data_range_diagnostics.txt`
    without removing any existing content.
    """
    # Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = Path(base_dir).parent
    final_csv_path = project_root / "final_options_dataset.csv"
    
    # Use different directory if filters are applied
    if APPLY_FILTERS:
        eda_dir = Path(base_dir) / "eda_all_figures_post_filtering"
        print(f"[Summary Stats] Using filtered output directory: {eda_dir}")
    else:
        eda_dir = Path(base_dir) / "eda_all_figures"
        
    eda_dir.mkdir(parents=True, exist_ok=True)
    out_csv_path = eda_dir / "summary_statistics.csv"
    out_txt_path = eda_dir / "data_range_diagnostics.txt"

    if not final_csv_path.exists():
        print(f"[Summary Stats] Final dataset not found at: {final_csv_path}")
        return

    print(f"[Summary Stats] Loading: {final_csv_path}")
    # Load efficiently: only numeric columns to compute stats correctly
    df = pd.read_csv(final_csv_path)
    
    # Apply filters if enabled
    if APPLY_FILTERS:
        print(f"[Summary Stats] Applying filters...")
        df = apply_data_filters(df)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("[Summary Stats] No numeric columns found; nothing to compute.")
        return

    # Prepare DataFrame for results
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors='coerce')
        # Use nan-aware computations
        n_non_missing = int(s.notna().sum())
        n_total = int(len(s))
        n_missing = int(n_total - n_non_missing)
        missing_pct = (n_missing / n_total * 100.0) if n_total > 0 else 0.0
        mean = float(np.nanmean(s))
        sd = float(np.nanstd(s, ddof=1))  # sample SD to match common stats outputs
        vmin = float(np.nanmin(s))
        q25 = float(np.nanpercentile(s, 25))
        med = float(np.nanpercentile(s, 50))
        q75 = float(np.nanpercentile(s, 75))
        vmax = float(np.nanmax(s))
        p1 = float(np.nanpercentile(s, 1))
        p99 = float(np.nanpercentile(s, 99))
        iqr = float(q75 - q25)
        cv = float(sd / mean) if np.isfinite(mean) and abs(mean) > 1e-12 and np.isfinite(sd) else np.nan
        # Skewness and Kurtosis (Pearson kurtosis, not excess)
        skew = float(stats.skew(s, nan_policy='omit')) if s.notna().any() else np.nan
        kurt = float(stats.kurtosis(s, fisher=False, nan_policy='omit')) if s.notna().any() else np.nan

        rows.append(dict(
            feature=col,
            Count=n_non_missing,
            Missing=n_missing,
            MissingPct=missing_pct,
            Mean=mean,
            SD=sd,
            Min=vmin,
            p25=q25,
            Median=med,
            p75=q75,
            Max=vmax,
            p1=p1,
            p99=p99,
            IQR=iqr,
            CV=cv,
            Skewness=skew,
            Kurtosis=kurt,
        ))

    summary_df = pd.DataFrame(rows).set_index('feature')
    # Sort index for consistency, but keep common key columns first if present
    priority = [
        'strike_price','best_bid','best_offer','volume','open_interest','mid_price',
        'days_to_maturity','historical_volatility','hist_vol_10d','hist_vol_30d','hist_vol_90d',
        'impl_volatility','risk_free_rate','dividend_rate',
        'spx_open','spx_high','spx_low','spx_close','moneyness',
        'epu_index','equity_uncertainty','equity_volatility'
    ]
    ordered_cols = [c for c in priority if c in summary_df.index] + [c for c in summary_df.index if c not in priority]
    summary_df = summary_df.loc[ordered_cols]

    # Save exportable CSV
    summary_df.to_csv(out_csv_path)
    print(f"[Summary Stats] Exported table to: {out_csv_path}")

    # Append to diagnostics text file without deleting anything
    header = (
        "\n" + "="*60 + "\n"
        "SUMMARY STATISTICS TABLE (Final Dataset)\n"
        + "="*60 + "\n\n"
    )

    # Create a formatted text table with aligned columns
    col_order = ['Count','Missing','MissingPct','Mean','SD','Min','p1','p25','Median','p75','p99','Max','IQR','CV','Skewness','Kurtosis']
    col_headers = "Feature".ljust(22) + " ".join(h.rjust(11) for h in col_order)

    lines = [header, col_headers, "-" * (22 + 12*len(col_order))]
    for feat, row in summary_df.iterrows():
        vals = [row[c] for c in col_order]
        # Format floats compactly; wide range values handled
        def fmt(x):
            if pd.isna(x):
                return "".rjust(11)
            # Choose dynamic precision
            if abs(x) >= 1000:
                return f"{x:11.2f}"
            elif abs(x) >= 1:
                return f"{x:11.3f}"
            else:
                return f"{x:11.4f}"
        line = feat.ljust(22) + "".join(fmt(v) for v in vals)
        lines.append(line)
    lines.append("\nNotes:\n")
    lines.append("- SD is sample standard deviation; Kurtosis shown is Pearson (not excess).\n")
    lines.append("- MissingPct is percentage of missing values. CV = SD/Mean.\n")

    # Ensure file exists; if not, create it before appending
    if not out_txt_path.exists():
        with open(out_txt_path, 'w') as f:
            f.write("DATA RANGE DIAGNOSTICS REPORT\n")

    with open(out_txt_path, 'a') as f:
        f.write("\n".join(lines))
    print(f"[Summary Stats] Appended summary table to: {out_txt_path}")

    # Additional: Save moneyness bucket distribution and append to diagnostics
    if 'moneyness' in df.columns:
        m = pd.to_numeric(df['moneyness'], errors='coerce')
        cats = pd.cut(m, bins=[0, 0.9, 1.1, np.inf], labels=['OTM','ATM','ITM'])
        bucket_counts = cats.value_counts().reindex(['OTM','ATM','ITM'])
        bucket_path = eda_dir / 'moneyness_bucket_counts.csv'
        bucket_counts.to_csv(bucket_path, header=['count'])
        print(f"[Summary Stats] Saved moneyness bucket counts to: {bucket_path}")
        with open(out_txt_path, 'a') as f:
            f.write("\nMONEYNESS (S/K) BUCKET COUNTS (OTM/ATM/ITM)\n")
            f.write("-"*40 + "\n")
            total = int(bucket_counts.sum()) if bucket_counts is not None else 0
            for lab in ['OTM','ATM','ITM']:
                cnt = int(bucket_counts.get(lab, 0))
                pct = (cnt/total*100.0) if total>0 else 0.0
                f.write(f"{lab}: {cnt:,} ({pct:.1f}%)\n")
            f.write("\n")

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
    if APPLY_FILTERS:
        print("--- DATA EXPLORER (WITH FILTERS) ---")
        print(f"[CONFIG] Filters: Zero Volume={ZERO_VOLUME_INCLUSION}, ITM={FILTER_ITM_OPTIONS}, "
              f"Moneyness=[{MONEYNESS_LOWER_BOUND}, {MONEYNESS_UPPER_BOUND}], "
              f"Days=[{MIN_DAYS_TO_MATURITY}, {MAX_DAYS_TO_MATURITY}], Valid Spread={FILTER_VALID_SPREAD}")
    else:
        print("--- DATA EXPLORER (NO FILTERS) ---")
        
    generate_summary_statistics_for_final_dataset()
    print(f"Script location: {Path(__file__).absolute()}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data path: {DATA_PATH}")
    print(f"Data path exists: {DATA_PATH.exists()}")
    explore_data()
