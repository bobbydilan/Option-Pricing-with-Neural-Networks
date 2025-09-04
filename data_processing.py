"""
Data Processing for Option Pricing MLP

This script contains functions to:
1. Load all raw data files
2. Match and merge auxiliary data (volatility, risk-free rate, dividend, SPX price)
3. Build final cleaned dataset for regression, Black Scholes and MLP training

Functions focus on data preparation only - no modeling here.
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
sns.set_palette("husl")

#%%

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(BASE_PATH, "00Data")
DEFAULT_ROW_LIMIT = 10000000

# CONFIGURATION: Choose processing method for large datasets
# Set to True for memory-efficient processing (slower but uses less RAM)
# Set to False for vectorized processing (faster but uses more RAM)
USE_LOOP_PROCESSING = True  # Change to True for large datasets ( > 8*10^5 rows)

#%%
def load_raw_data(data_path=DATA_PATH, row_limit=DEFAULT_ROW_LIMIT):

    print(f"[INFO] Loading data from: {data_path}")

    filenames = {
        'options': 'data1.csv',
        'hist_vol': 'Historical Volatility of SP500.csv',
        'risk_free': 'Continuously compounded, risk-free interest rate.csv',
        'spx_price': 'SPX Price.csv',
        'spx_dividend': 'Continuous SP500 Dividend Yield.csv',
        'epu_index': 'Economic Policy Uncertainty Index for United States.csv',
        'equity_uncertainty': 'Equity Market Economic Uncertainty Index.csv',
        'equity_volatility': 'Equity Market Volatility.csv',
    }
    data = {}
    for key, fname in filenames.items():
        file_path = os.path.join(data_path, fname)
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            data[key] = None
        else:
            print(f"[INFO] Loading {key} from {fname} ...")
            data[key] = pd.read_csv(file_path)
            if key == 'options' and row_limit is not None:
                data[key] = data[key].head(row_limit)
                print(f"[INFO] Trimmed 'options' data to first {row_limit} rows for testing")
            print(f"[INFO] {key} shape: {data[key].shape}")
    print("[INFO] Raw data loading complete.")
    return data

#%%
def visualize_raw_data(data):
    
    options_df = data['options']
    print("Creating comprehensive data visualizations...")
    
    print("\n" + "="*60)
    print("ENHANCED DATA SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\n OPTIONS DATA:")
    print(f"   Total options: {len(options_df):,}")
    print(f"   Calls: {len(options_df[options_df['cp_flag'] == 'C']):,}")
    print(f"   Puts: {len(options_df[options_df['cp_flag'] == 'P']):,}")
    print(f"   Date range: {options_df['date'].min()} to {options_df['date'].max()}")
    print(f"   Strike range: ${options_df['strike_price'].min():.2f} - ${options_df['strike_price'].max():.2f}")
    print(f"   Avg implied volatility: {options_df['impl_volatility'].mean():.3f}")
    
    print(f"\n  SPX PRICE DATA:")
    spx_summary = data['spx_price']
    print(f"   Records: {len(spx_summary):,}")
    print(f"   Price range: ${spx_summary['close'].min():.2f} - ${spx_summary['close'].max():.2f}")
    
    print(f"\n  RISK-FREE RATE DATA:")
    rf_summary = data['risk_free']
    print(f"   Records: {len(rf_summary):,}")
    print(f"   Rate range: {rf_summary['rate'].min():.3f}% - {rf_summary['rate'].max():.3f}%")
    print(f"   Maturity range: {rf_summary['days'].min()} - {rf_summary['days'].max()} days")
    
    print(f"\n  HISTORICAL VOLATILITY DATA:")
    vol_summary = data['hist_vol']
    print(f"   Records: {len(vol_summary):,}")
    print(f"   Volatility range: {vol_summary['volatility'].min():.3f} - {vol_summary['volatility'].max():.3f}")
    
    print(f"\n  DIVIDEND DATA:")
    div_summary = data['spx_dividend']
    print(f"   Records: {len(div_summary):,}")
    print(f"   Rate range: {div_summary['rate'].min():.3f}% - {div_summary['rate'].max():.3f}%")

#%%
def prepare_options_data(options_df):
    #Prepare and clean the main options dataset
    print("Preparing options data...")
    
    options_clean = options_df.copy()
    print(f"Processing {len(options_clean):,} options (calls and puts)")
    
    options_clean['date'] = pd.to_datetime(options_clean['date'])
    options_clean['exdate'] = pd.to_datetime(options_clean['exdate'])
    
    # Convert strike price from contract notation to per-share basis. Options contracts represent 1000 shares, so divide by 1000
    options_clean['strike_price'] = options_clean['strike_price']/1000
    
    options_clean['days_to_maturity'] = (options_clean['exdate'] - options_clean['date']).dt.days
    options_clean['mid_price'] = (options_clean['best_bid'] + options_clean['best_offer']) / 2
    
    options_clean = options_clean[
        (options_clean['days_to_maturity'] >= 0) &
        (options_clean['best_bid'] > 0) &
        (options_clean['best_offer'] > 0) &
        (options_clean['strike_price'] > 0)
    ]
    
    cp_counts = options_clean['cp_flag'].value_counts()
    print(f"After cleaning: {len(options_clean):,} valid options")
    print(f"  - Calls: {cp_counts.get('C', 0):,}")
    print(f"  - Puts: {cp_counts.get('P', 0):,}")
    
    return options_clean

def _prepare_data_for_matching(options_df, aux_df, result_column):
    """Prepare options and auxiliary data for matching operations"""
    aux_work = aux_df.copy()
    aux_work['date'] = pd.to_datetime(aux_work['date'])
    
    options_work = options_df.copy()
    options_work['date'] = pd.to_datetime(options_work['date'])
    options_work[result_column] = np.nan
    
    return options_work, aux_work

def _group_aux_data_by_date(aux_df, days_col, value_col):
    """Group auxiliary data by date for efficient lookup"""
    data_by_date = {}
    for _, row in aux_df.iterrows():
        date = row['date']
        if date not in data_by_date:
            data_by_date[date] = []
        data_by_date[date].append((row[days_col], row[value_col]))
    return data_by_date

def _process_options_with_progress(options_work, data_by_date, result_column, value_transform=None):
    """Process options with progress tracking and find closest matches"""
    matched_count = 0
    total_options = len(options_work)
    
    for idx, option_row in options_work.iterrows():
        if idx % 200000 == 0:
            print(f"  Processed {idx:,}/{total_options:,} options ({idx/total_options*100:.1f}%)")
        
        option_date = option_row['date']
        option_ttm = option_row['days_to_maturity']
        
        if option_date in data_by_date:
            aux_data = data_by_date[option_date]
            if aux_data:
                best_match = min(aux_data, key=lambda x: abs(x[0] - option_ttm))
                value = best_match[1]
                if value_transform:
                    value = value_transform(value)
                if value is not None:
                    options_work.at[idx, result_column] = value
                    matched_count += 1
    
    return options_work, matched_count

def match_historical_volatility(options_df, hist_vol_df):
    """
    Match historical volatility to options based on date and days to maturity.
    Each option gets:
    1. Historical volatility with days closest to its specific days to maturity
    2. Specific 10-day, 30-day, and 90-day historical volatilities for that date
    """
    print("Matching historical volatility with specific periods (10d, 30d, 90d) and closest TTM match...")
    
    vol_df = hist_vol_df.copy()
    vol_df['date'] = pd.to_datetime(vol_df['date'])
    
    options_work = options_df.copy()
    options_work['date'] = pd.to_datetime(options_work['date'])
    
    # Initialize all volatility columns
    options_work['historical_volatility'] = np.nan  # Closest match to each option's TTM
    options_work['hist_vol_10d'] = np.nan          # Exactly 10d vol for that date
    options_work['hist_vol_30d'] = np.nan          # Exactly 30d vol for that date  
    options_work['hist_vol_90d'] = np.nan          # Exactly 90d vol for that date
    
    vol_grouped = vol_df.groupby('date')
    matched_count = 0
    vol_10d_count = 0
    vol_30d_count = 0
    vol_90d_count = 0
    
    for idx, row in options_work.iterrows():
        date, ttm = row['date'], row['days_to_maturity']
        
        if date in vol_grouped.groups:
            group = vol_grouped.get_group(date)
            days_vals = group['days'].values
            vol_vals = group['volatility'].values
            
            # 1. CLOSEST MATCH: Find volatility closest to this option's specific TTM
            closest_idx = np.argmin(np.abs(days_vals - ttm))
            options_work.at[idx, 'historical_volatility'] = vol_vals[closest_idx]
            matched_count += 1
            
            # 2. SPECIFIC PERIODS: Get exact 10d, 30d, 90d volatilities for this date
            target_periods = [
                (10, 'hist_vol_10d'),
                (30, 'hist_vol_30d'), 
                (90, 'hist_vol_90d')  # Look for 90d, but accept 91d if close
            ]
            
            for target_days, col_name in target_periods:
                # Find closest match to target period
                period_distances = np.abs(days_vals - target_days)
                period_closest_idx = np.argmin(period_distances)
                closest_period = days_vals[period_closest_idx]
                closest_distance = period_distances[period_closest_idx]
                
                # Set tolerance based on target period
                if target_days == 10:
                    tolerance = 5   # ±5 days for 10d
                elif target_days == 30:
                    tolerance = 10  # ±10 days for 30d
                else:  # 90d
                    tolerance = 15  # ±15 days for 90d (to catch 91d)
                
                # Only assign if within tolerance
                if closest_distance <= tolerance:
                    options_work.at[idx, col_name] = vol_vals[period_closest_idx]
                    
                    # Count successful matches
                    if target_days == 10:
                        vol_10d_count += 1
                    elif target_days == 30:
                        vol_30d_count += 1
                    elif target_days == 90:
                        vol_90d_count += 1
    
    # Summary statistics
    total_options = len(options_work)
    print(f"Processed {total_options:,} total options")
    print(f"Matched closest TTM volatility for {matched_count:,} options ({matched_count/total_options*100:.1f}%)")
    print(f"Matched 10-day volatility for {vol_10d_count:,} options ({vol_10d_count/total_options*100:.1f}%)")
    print(f"Matched 30-day volatility for {vol_30d_count:,} options ({vol_30d_count/total_options*100:.1f}%)")
    print(f"Matched 90-day volatility for {vol_90d_count:,} options ({vol_90d_count/total_options*100:.1f}%)")
    
    return options_work


def match_risk_free_rate(options_df, risk_free_df):
    rf_df = risk_free_df.copy()
    rf_df['date'] = pd.to_datetime(rf_df['date'])
    options_work = options_df.copy()
    options_work['date'] = pd.to_datetime(options_work['date'])
    options_work['orig_idx'] = options_work.index
    merged = options_work.merge(rf_df, on='date', how='left')
    merged['days_diff'] = np.abs(merged['days'] - merged['days_to_maturity'])
    idx_min = merged.groupby('orig_idx')['days_diff'].idxmin()
    result_df = options_work.copy()
    result_df['risk_free_rate'] = np.nan
    if len(idx_min) > 0:
        matched = merged.loc[idx_min, ['orig_idx', 'rate']].copy()
        matched['risk_free_rate'] = np.where(matched['rate'] > 0, matched['rate'] / 100.0, np.nan)
        result_df.loc[matched['orig_idx'], 'risk_free_rate'] = matched['risk_free_rate'].values
    del merged, rf_df, options_work
    
    print(f"Matched risk-free rate for {(~result_df['risk_free_rate'].isna()).sum():,} options")
    return result_df

def match_spx_price(options_df, spx_price_df):
    spx_df = spx_price_df.copy()
    spx_df['date'] = pd.to_datetime(spx_df['date'])
    result_df = pd.merge(
        options_df,
        spx_df[['date', 'secid', 'open', 'high', 'low', 'close']],
        on=['date', 'secid'],
        how='left',
    )
    result_df = result_df.rename(columns={
        'open': 'spx_open',
        'high': 'spx_high',
        'low': 'spx_low',
        'close': 'spx_close'
    })
    del spx_df
    print(f"Matched SPX price for {(~result_df['spx_close'].isna()).sum():,} options")
    return result_df

def match_dividend_data(options_df, dividend_df):
    div_df = dividend_df.copy()
    div_df['date'] = pd.to_datetime(div_df['date'])
    div_df['expiration'] = pd.to_datetime(div_df['expiration'], errors='coerce', dayfirst=False)
    if div_df['expiration'].isna().sum() > 0.5 * len(div_df):
        div_df['expiration'] = pd.to_datetime(div_df['expiration'], errors='coerce', dayfirst=True)
    div_df = div_df.dropna(subset=['expiration'])
    div_df['days_to_div_exp'] = (div_df['expiration'] - div_df['date']).dt.days
    options_work = options_df.copy()
    options_work['date'] = pd.to_datetime(options_work['date'])
    options_work['exdate'] = pd.to_datetime(options_work['exdate'])
    options_work['days_to_maturity'] = (options_work['exdate'] - options_work['date']).dt.days
    options_work['orig_idx'] = options_work.index
    merged = options_work.merge(div_df, on='date', how='left')
    merged['exp_diff'] = np.abs(merged['days_to_div_exp'] - merged['days_to_maturity'])
    idx_min = merged.groupby('orig_idx')['exp_diff'].idxmin()
    result_df = options_work.copy()
    result_df['dividend_rate'] = np.nan
    if len(idx_min) > 0:
        matched = merged.loc[idx_min, ['orig_idx', 'rate']].copy()
        matched['dividend_rate'] = matched['rate'] / 100.0
        result_df.loc[matched['orig_idx'], 'dividend_rate'] = matched['dividend_rate'].values
    del merged, div_df, options_work
    print(f"Matched dividend rate for {(~result_df['dividend_rate'].isna()).sum():,} options")
    return result_df

#%%
def match_epu_index(options_df, epu_df):
    # Match Economic Policy Uncertainty Index to options based on date (optimized)
    
    print("Matching Economic Policy Uncertainty Index...")
    result_df = options_df.copy()
    result_df['epu_index'] = np.nan
    
    epu_work = epu_df.copy()
    epu_work['observation_date'] = pd.to_datetime(epu_work['observation_date'])
    epu_work = epu_work.sort_values('observation_date')
    
    # Get unique option dates and their indices
    unique_dates = result_df['date'].unique()
    print(f"Processing {len(unique_dates):,} unique option dates...")
    
    # Process each unique date once
    for option_date in unique_dates:
        # Find closest EPU date (backward looking - use most recent available data)
        epu_subset = epu_work[epu_work['observation_date'] <= option_date]
        if len(epu_subset) > 0:
            closest_epu_value = epu_subset.iloc[-1]['USEPUINDXD']  # Most recent
            # Apply to all options with this date
            result_df.loc[result_df['date'] == option_date, 'epu_index'] = closest_epu_value
    
    print(f"Matched EPU index for {(~result_df['epu_index'].isna()).sum():,} options")
    return result_df

#%%
def match_equity_uncertainty(options_df, equity_unc_df):
    # Match Equity Market Economic Uncertainty Index to options based on date (optimized)
    
    result_df = options_df.copy()
    result_df['equity_uncertainty'] = np.nan
    
    # Prepare equity uncertainty data
    equity_unc_work = equity_unc_df.copy()
    equity_unc_work['observation_date'] = pd.to_datetime(equity_unc_work['observation_date'])
    equity_unc_work = equity_unc_work.sort_values('observation_date')
    
    # Get unique option dates and their indices
    unique_dates = result_df['date'].unique()
    print(f"Processing {len(unique_dates):,} unique option dates...")
    
    # Process each unique date once
    for option_date in unique_dates:
        # Find closest equity uncertainty date (backward looking)
        unc_subset = equity_unc_work[equity_unc_work['observation_date'] <= option_date]
        if len(unc_subset) > 0:
            closest_unc_value = unc_subset.iloc[-1]['WLEMUINDXD']  # Most recent
            # Apply to all options with this date
            result_df.loc[result_df['date'] == option_date, 'equity_uncertainty'] = closest_unc_value
    
    print(f"Matched equity uncertainty for {(~result_df['equity_uncertainty'].isna()).sum():,} options")
    return result_df

#%%
def match_equity_volatility(options_df, equity_vol_df):
    # Match Equity Market Volatility to options based on date (optimized)

    result_df = options_df.copy()
    result_df['equity_volatility'] = np.nan
    
    # Prepare equity volatility data
    equity_vol_work = equity_vol_df.copy()
    equity_vol_work['observation_date'] = pd.to_datetime(equity_vol_work['observation_date'])
    equity_vol_work = equity_vol_work.sort_values('observation_date')
    
    # Get unique option dates and their indices
    unique_dates = result_df['date'].unique()
    print(f"Processing {len(unique_dates):,} unique option dates...")
    
    # Process each unique date once
    for option_date in unique_dates:
        # Find closest equity volatility date (backward looking)
        vol_subset = equity_vol_work[equity_vol_work['observation_date'] <= option_date]
        if len(vol_subset) > 0:
            closest_vol_value = vol_subset.iloc[-1]['INFECTDISEMVTRACKD']  # Most recent
            # Apply to all options with this date
            result_df.loc[result_df['date'] == option_date, 'equity_volatility'] = closest_vol_value
    
    print(f"Matched equity volatility for {(~result_df['equity_volatility'].isna()).sum():,} options")
    return result_df

#%%
# MEMORY-EFFICIENT LOOP-BASED ALTERNATIVES FOR LARGE DATASETS
# These functions use for loops instead of vectorization to reduce memory usage

def match_historical_volatility_loop(options_df, hist_vol_df):
    """Match historical volatility using memory-efficient for loops with specific periods"""
    print("Matching historical volatility with specific periods using memory-efficient for loops...")
    
    vol_df = hist_vol_df.copy()
    vol_df['date'] = pd.to_datetime(vol_df['date'])
    
    options_work = options_df.copy()
    options_work['date'] = pd.to_datetime(options_work['date'])
    
    # Initialize all volatility columns
    options_work['historical_volatility'] = np.nan
    options_work['hist_vol_10d'] = np.nan
    options_work['hist_vol_30d'] = np.nan
    options_work['hist_vol_90d'] = np.nan
    
    # Group volatility data by date for efficient lookup
    vol_by_date = {}
    for _, row in vol_df.iterrows():
        date = row['date']
        if date not in vol_by_date:
            vol_by_date[date] = []
        vol_by_date[date].append((row['days'], row['volatility']))
    
    # Process options with progress tracking
    matched_count = 0
    vol_10d_count = 0
    vol_30d_count = 0
    vol_90d_count = 0
    total_options = len(options_work)
    
    for idx, option_row in options_work.iterrows():
        if idx % 200000 == 0:
            print(f"  Processed {idx:,}/{total_options:,} options ({idx/total_options*100:.1f}%)")
        
        option_date = option_row['date']
        option_ttm = option_row['days_to_maturity']
        
        if option_date in vol_by_date:
            vol_data = vol_by_date[option_date]
            if vol_data:
                # Match closest volatility to days to maturity
                best_match = min(vol_data, key=lambda x: abs(x[0] - option_ttm))
                options_work.at[idx, 'historical_volatility'] = best_match[1]
                matched_count += 1
                
                # Match specific periods (10d, 30d, 90d)
                for target_days, col_name in [(10, 'hist_vol_10d'), (30, 'hist_vol_30d'), (91, 'hist_vol_90d')]:
                    closest_match = min(vol_data, key=lambda x: abs(x[0] - target_days))
                    closest_days = closest_match[0]
                    
                    # Only use if reasonably close
                    tolerance = 5 if target_days == 10 else 10
                    if abs(closest_days - target_days) <= tolerance:
                        options_work.at[idx, col_name] = closest_match[1]
                        if target_days == 10:
                            vol_10d_count += 15
                        elif target_days == 30:
                            vol_30d_count += 15
                        elif target_days == 91:
                            vol_90d_count += 15
    
    print(f"Matched closest volatility for {matched_count:,} options")
    print(f"Matched 10-day volatility for {vol_10d_count:,} options")
    print(f"Matched 30-day volatility for {vol_30d_count:,} options")
    print(f"Matched 90-day volatility for {vol_90d_count:,} options")
    return options_work

def match_risk_free_rate_loop(options_df, risk_free_df):
    # Match risk-free rate using memory-efficient for loops
    
    print("Matching risk-free rate using memory-efficient for loops...")
    options_work, rf_df = _prepare_data_for_matching(options_df, risk_free_df, 'risk_free_rate')
    rf_by_date = _group_aux_data_by_date(rf_df, 'days', 'rate')
    
    def rate_transform(rate):
        return rate / 100.0 if rate > 0 else None
    
    options_work, matched_count = _process_options_with_progress(options_work, rf_by_date, 'risk_free_rate', rate_transform)
    print(f"Matched risk-free rate for {matched_count:,} options")
    return options_work

def _prepare_dividend_data(dividend_df):
    # Prepare dividend data with expiration date processing
    
    div_df = dividend_df.copy()
    div_df['date'] = pd.to_datetime(div_df['date'])
    div_df['expiration'] = pd.to_datetime(div_df['expiration'], errors='coerce', dayfirst=False)
    div_df = div_df.dropna(subset=['expiration'])
    div_df['days_to_div_exp'] = (div_df['expiration'] - div_df['date']).dt.days
    return div_df

def match_dividend_data_loop(options_df, dividend_df):
    # Match dividend data using memory-efficient for loops
    print("Matching dividend data using memory-efficient for loops...")
    
    div_df = _prepare_dividend_data(dividend_df)
    
    # Prepare options data with days_to_maturity calculation
    options_work = options_df.copy()
    options_work['date'] = pd.to_datetime(options_work['date'])
    options_work['exdate'] = pd.to_datetime(options_work['exdate'])
    options_work['days_to_maturity'] = (options_work['exdate'] - options_work['date']).dt.days
    options_work['dividend_rate'] = np.nan
    
    div_by_date = _group_aux_data_by_date(div_df, 'days_to_div_exp', 'rate')
    
    def dividend_transform(rate):
        return rate / 100.0  # Convert to decimal
    
    options_work, matched_count = _process_options_with_progress(options_work, div_by_date, 'dividend_rate', dividend_transform)
    print(f"Matched dividend rate for {matched_count:,} options")
    return options_work

def build_final_dataset(data_path=DATA_PATH):
    raw_data = load_raw_data(data_path)
    options_df = prepare_options_data(raw_data['options'])
    
    if USE_LOOP_PROCESSING:
        print("\n[INFO] Using memory-efficient loop-based processing for large datasets...")
        options_df = match_historical_volatility_loop(options_df, raw_data['hist_vol'])
        options_df = match_risk_free_rate_loop(options_df, raw_data['risk_free'])
        options_df = match_spx_price(options_df, raw_data['spx_price'])  # SPX matching is already efficient
        options_df = match_dividend_data_loop(options_df, raw_data['spx_dividend'])
    
    else:
        print("\n[INFO] Using vectorized processing (faster but more memory-intensive)...")
        options_df = match_historical_volatility(options_df, raw_data['hist_vol'])
        options_df = match_risk_free_rate(options_df, raw_data['risk_free'])
        options_df = match_spx_price(options_df, raw_data['spx_price'])
        options_df = match_dividend_data(options_df, raw_data['spx_dividend'])
    
    # Match the new economic/uncertainty data (using loop-based approach for all)
    print("\n[INFO] Matching economic and uncertainty indices...")
    options_df = match_epu_index(options_df, raw_data['epu_index'])
    options_df = match_equity_uncertainty(options_df, raw_data['equity_uncertainty'])
    options_df = match_equity_volatility(options_df, raw_data['equity_volatility'])
    options_df['moneyness'] = options_df['strike_price'] / options_df['spx_close']
    final_df = options_df.copy()
    
    # Final quality checks. Remove rows with negative risk-free rate or dividend rate
    final_df = final_df[(final_df['risk_free_rate'] >= 0) & (final_df['dividend_rate'] >= 0)]

    print(f"Final dataset size after cleaning: {len(final_df):,}")

    if len(final_df) == 0:
        print("[ERROR] Final dataset is empty after filtering. Please check the previous cleaning steps and input data.")

    # Select final columns for the model
    final_columns = [
        'date', 'secid', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 
        'volume', 'open_interest', 'mid_price', 'days_to_maturity', 'historical_volatility',
        'hist_vol_10d', 'hist_vol_30d', 'hist_vol_90d', 'impl_volatility', 'risk_free_rate', 'dividend_rate', 'spx_open', 'spx_high', 
        'spx_low', 'spx_close', 'moneyness', 'epu_index', 'equity_uncertainty', 'equity_volatility'
    ]
    
    final_df = final_df[final_columns]
    
    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Final columns: {list(final_df.columns)}")
    
    return final_df.dropna(subset=['risk_free_rate', 'date', 'strike_price','exdate', 'days_to_maturity', 'historical_volatility', 'dividend_rate', 'spx_close', 'mid_price'])

#%%
if __name__ == "__main__":
    final_dataset = build_final_dataset()
    
    output_path = os.path.join(BASE_PATH, "final_options_dataset.csv")
    final_dataset.to_csv(output_path, index=False)
    print(f"\nFinal dataset saved to '{output_path}'")

    print("\n" + "="*50)
    print("FINAL DATASET SUMMARY")
    print("="*50)
    print(final_dataset.describe())
