import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

SEED = 42
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
VAL_TEST_SPLIT = VALIDATION_RATIO / (1 - TRAIN_RATIO)
_TEMP_TOTAL = max(1e-12, 1.0 - TRAIN_RATIO)
VAL_FRACTION_OF_TEMP = VALIDATION_RATIO / _TEMP_TOTAL
TEST_FRACTION_OF_TEMP = 1.0 - VAL_FRACTION_OF_TEMP
USE_TIME_BASED_SPLIT = True
USE_TIME_PERCENT_SPLIT = True
TIME_SPLIT_FRACTION = TRAIN_RATIO
USE_MULTI_DIM_STRATIFICATION = False
SAMPLE_SIZE = 10000000
ZERO_VOLUME_INCLUSION = 0.5
MONEYNESS_LOWER_BOUND = 0.5
MONEYNESS_UPPER_BOUND = 1.5
MIN_DAYS_TO_MATURITY = 0
MAX_DAYS_TO_MATURITY = 750
FILTER_VALID_SPREAD = False
SPREAD_FILTER_ENABLED = False
MIN_SPREAD_PCT = None
MAX_SPREAD_PCT = None

# Feature Selection
BASE_FEATURES = ['strike_price', 'historical_volatility', 'dividend_rate', 'risk_free_rate', 'days_to_maturity', 'spx_close']
ADDITIONAL_RAW_FEATURES = ['volume', 'hist_vol_10d', 'hist_vol_30d', 'hist_vol_90d', 'spx_open', 'spx_high', 'spx_low', 'moneyness', 'open_interest', 'epu_index', 'equity_uncertainty', 'equity_volatility']

'''
'volume', 'hist_vol_10d', 'hist_vol_30d', 'hist_vol_90d', 'spx_open', 'spx_high', 'spx_low', 'moneyness', 'open_interest', 'epu_index', 'equity_uncertainty', 'equity_volatility
'''

# Model Parameters
POLY_DEGREE = 2
INCLUDE_INTERACTION_TERMS = False

SELECTED_FEATURES = [
    'strike_price','historical_volatility','dividend_rate', 'risk_free_rate', 'days_to_maturity', 'spx_close']  #, 'volume', 'open_interest', 'epu_index'

RESULTS_FOLDER = 'results'
USE_TIMESTAMP = True

MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
MONEYNESS_BIN_LABELS = ['OTM\n(<0.9)', 'ATM\n(0.9-1.1)', 'ITM\n(>1.1)']
TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]
TIME_BIN_LABELS = ['≤1M\n(≤30d)', '1-3M\n(31-91d)', '3-6M\n(92-182d)', '6-9M\n(183-274d)', '9-12M\n(275-365d)', '>12M\n(>365d)']
MONEYNESS_BINS = np.linspace(0.7, 1.3, 13)
TIME_BINS = np.linspace(0, 2, 9)
MONEYNESS_LABELS = [f'{MONEYNESS_BINS[i]:.2f}-{MONEYNESS_BINS[i+1]:.2f}' for i in range(len(MONEYNESS_BINS)-1)]
TIME_LABELS = [f'{TIME_BINS[i]:.2f}-{TIME_BINS[i+1]:.2f}' for i in range(len(TIME_BINS)-1)]

def setup_plot_style():
    """Set up consistent plot styling to avoid repetition."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5
    })

def get_volume_buckets(volume):
    """Create volume buckets for analysis"""
    volume_bins = [-0.1, 0, 100, 1000, np.inf]
    volume_labels = ['0', '0-100', '100-1000', '1000+']
    return pd.cut(volume, bins=volume_bins, labels=volume_labels, include_lowest=True)

def get_hv_buckets(hv_series: pd.Series) -> pd.Categorical:
    """Create historical volatility quantile buckets for analysis"""
    return pd.qcut(hv_series, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

def get_price_buckets(price_series: pd.Series) -> pd.Categorical:
    """Create price quantile buckets for analysis"""
    return pd.qcut(price_series, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

def get_uncertainty_buckets(uncertainty_series: pd.Series) -> pd.Categorical:
    """Create equity uncertainty buckets for analysis"""
    uncertainty_bins = [0, 50, 100, 150, np.inf]
    uncertainty_labels = ['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)']
    return pd.cut(uncertainty_series, bins=uncertainty_bins, labels=uncertainty_labels, include_lowest=True)

def create_bucket_bar_plot(ax, analysis_df, metric_col, count_col, title, ylabel, color, rotation=0):
    """Reusable function for creating bucket analysis bar plots"""
    if analysis_df.empty:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        return
    
    bars = ax.bar(range(len(analysis_df)), analysis_df[metric_col], 
                  color=color, alpha=0.8, edgecolor='black')
    
    for i, (bar, count) in enumerate(zip(bars, analysis_df[count_col])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(analysis_df[metric_col]) * 0.01,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(range(len(analysis_df)))
    ax.set_xticklabels(analysis_df.index, rotation=rotation, ha='right' if rotation > 0 else 'center')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)

def get_moneyness_buckets(moneyness_series: pd.Series) -> pd.Categorical:
    return pd.cut(moneyness_series, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)

def get_time_buckets(time_years_series: pd.Series) -> pd.Categorical:
    return pd.cut(time_years_series, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)

def perform_bucket_analysis(df_test, results_dir):
    """Perform comprehensive bucket analysis across option characteristics"""
    print("\n=== PERFORMING BUCKET ANALYSIS ===")
    
    if len(df_test) == 0 or 'price_error' not in df_test.columns:
        print("Warning: Missing required columns for bucket analysis")
        return
    
    analysis_results = {}
    
    # Moneyness Analysis
    if 'moneyness' in df_test.columns:
        df_analysis = df_test.copy()
        df_analysis['moneyness_bucket'] = get_moneyness_buckets(df_analysis['moneyness'])
        analysis_results['moneyness'] = df_analysis.groupby('moneyness_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        analysis_results['moneyness'].columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        analysis_results['moneyness'] = analysis_results['moneyness'][['Count', 'MAE', 'MAPE', 'Median', 'SD']]
    
    # Time to Expiration Analysis
    if 'days_to_maturity' in df_test.columns:
        df_analysis = df_test.copy()
        df_analysis['tte_bucket'] = get_time_buckets(df_analysis['days_to_maturity'] / 365.25)
        analysis_results['time_to_expiration'] = df_analysis.groupby('tte_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        analysis_results['time_to_expiration'].columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        analysis_results['time_to_expiration'] = analysis_results['time_to_expiration'][['Count', 'MAE', 'MAPE', 'Median', 'SD']]
    
    # Volume Analysis
    if 'volume' in df_test.columns:
        df_analysis = df_test.copy()
        df_analysis['volume_bucket'] = get_volume_buckets(df_analysis['volume'])
        analysis_results['volume'] = df_analysis.groupby('volume_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        analysis_results['volume'].columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        analysis_results['volume'] = analysis_results['volume'][['Count', 'MAE', 'MAPE', 'Median', 'SD']]
    
    # Historical Volatility Analysis
    if 'historical_volatility' in df_test.columns:
        df_analysis = df_test.copy()
        df_analysis['historical_volatility_bucket'] = get_hv_buckets(df_analysis['historical_volatility'])
        
        # Get quantile ranges for display
        hv_quantiles = df_analysis['historical_volatility'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        hv_ranges = {}
        for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
            min_val = hv_quantiles.iloc[i]
            max_val = hv_quantiles.iloc[i+1]
            hv_ranges[q] = f"Q{i+1} ({min_val:.3f} - {max_val:.3f})"
        
        hv_analysis = df_analysis.groupby('historical_volatility_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        hv_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        hv_analysis = hv_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
        
        # Rename index with ranges
        hv_analysis.index = [hv_ranges.get(str(idx), str(idx)) for idx in hv_analysis.index]
        analysis_results['historical_volatility'] = hv_analysis
    
    # Price Range Analysis
    if 'mid_price' in df_test.columns:
        df_analysis = df_test.copy()
        df_analysis['price_bucket'] = get_price_buckets(df_analysis['mid_price'])
        
        # Get quantile ranges for display
        price_quantiles = df_analysis['mid_price'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        price_ranges = {}
        for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
            min_val = price_quantiles.iloc[i]
            max_val = price_quantiles.iloc[i+1]
            price_ranges[q] = f"Q{i+1} (${min_val:.2f} - ${max_val:.2f})"
        
        price_analysis = df_analysis.groupby('price_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        price_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        price_analysis = price_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
        
        # Rename index with ranges
        price_analysis.index = [price_ranges.get(str(idx), str(idx)) for idx in price_analysis.index]
        analysis_results['price_range'] = price_analysis
    
    # Equity Uncertainty Analysis
    if 'equity_uncertainty' in df_test.columns:
        df_analysis = df_test.copy()
        df_analysis['uncertainty_bucket'] = get_uncertainty_buckets(df_analysis['equity_uncertainty'])
        analysis_results['equity_uncertainty'] = df_analysis.groupby('uncertainty_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        analysis_results['equity_uncertainty'].columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        analysis_results['equity_uncertainty'] = analysis_results['equity_uncertainty'][['Count', 'MAE', 'MAPE', 'Median', 'SD']]
    
    print("=== BUCKET ANALYSIS COMPLETE ===\n")
    return analysis_results

def create_data_splits(X, y, df, indices):
    if USE_TIME_BASED_SPLIT and 'date' in df.columns:
        if USE_TIME_PERCENT_SPLIT:
            # Percentage-based chronological split
            print(f"Using time-based split by fraction: first {TIME_SPLIT_FRACTION:.2%} of current-order data as train")

            positions = np.arange(len(df))
            n = len(positions)
            cutoff = int(np.floor(TIME_SPLIT_FRACTION * n))
            cutoff = max(1, min(cutoff, n - 1))  # ensure non-empty train and temp

            train_positions = positions[:cutoff]
            temp_positions = positions[cutoff:]

            X_train, y_train = X[train_positions], y[train_positions]
            X_temp, y_temp = X[temp_positions], y[temp_positions]

            train_indices = indices[train_positions]
            temp_indices = indices[temp_positions]

            # Split temp into validation and test (random; optionally stratified)
            if len(temp_indices) > 0:
                temp_df = df.iloc[temp_positions].reset_index(drop=True)
                temp_X_indices = np.arange(len(X_temp))

                if USE_MULTI_DIM_STRATIFICATION:
                    temp_strat = _build_multidim_strata(temp_df)
                else:
                    temp_strat = temp_df['mid_price'].values if 'mid_price' in temp_df.columns else None

                if temp_strat is not None and np.issubdtype(np.array(temp_strat).dtype, np.number) and len(np.unique(temp_strat)) > 20:
                    temp_strat = pd.qcut(temp_strat, q=3, labels=False, duplicates='drop')

                X_val, X_test, y_val, y_test, temp_idx_val, temp_idx_test = train_test_split(
                    X_temp, y_temp, temp_X_indices, test_size=VAL_TEST_SPLIT, random_state=SEED,
                    stratify=temp_strat if temp_strat is not None else None)

                # Map back to original indices
                idx_train = train_indices
                idx_val = temp_indices[temp_idx_val]
                idx_test = temp_indices[temp_idx_test]
            else:
                print("Warning: No data after time fraction cutoff, falling back to random split")
                return create_random_splits(X, y, indices, df)

        else:
            print(f"Using time-based split with cutoff date: {TIME_SPLIT_DATE}")
            
            # Convert split date to datetime
            split_date = pd.to_datetime(TIME_SPLIT_DATE)
            
            # Create time-based splits
            train_mask = df['date'] < split_date
            test_val_mask = df['date'] >= split_date
            
            # Get indices for train and temp (val+test)
            train_indices = indices[train_mask]
            temp_indices = indices[test_val_mask]
            
            X_train, y_train = X[train_mask], y[train_mask]
            X_temp, y_temp = X[test_val_mask], y[test_val_mask]
            
            # Split the remaining data into validation and test
            if len(temp_indices) > 0:
                temp_df = df[test_val_mask].reset_index(drop=True)
                temp_X_indices = np.arange(len(X_temp))
                
                if USE_MULTI_DIM_STRATIFICATION:
                    temp_strat = _build_multidim_strata(temp_df)
                else:
                    temp_strat = temp_df['mid_price'].values if 'mid_price' in temp_df.columns else None

                if temp_strat is not None and np.issubdtype(np.array(temp_strat).dtype, np.number) and len(np.unique(temp_strat)) > 20:
                    temp_strat = pd.qcut(temp_strat, q=3, labels=False, duplicates='drop')

                X_val, X_test, y_val, y_test, temp_idx_val, temp_idx_test = train_test_split(
                    X_temp, y_temp, temp_X_indices, test_size=VAL_TEST_SPLIT, random_state=SEED,
                    stratify=temp_strat if temp_strat is not None else None)
                
                # Map back to original indices
                idx_train = train_indices
                idx_val = temp_indices[temp_idx_val]
                idx_test = temp_indices[temp_idx_test]
            else:
                # Fallback if no data after split date
                print("Warning: No data after split date, falling back to random split")
                return create_random_splits(X, y, indices, df)
            
    else:
        print("Using improved random split with stratification")
        return create_random_splits(X, y, indices, df)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test

def create_random_splits(X, y, indices, df):
    if USE_MULTI_DIM_STRATIFICATION:
        # Create stratification labels
        df_subset = df.iloc[indices] if indices is not None else df
        stratification_labels = _build_multidim_strata(df_subset)
        
        # Check if we have enough samples per stratum
        label_counts = pd.Series(stratification_labels).value_counts()
        min_samples_per_stratum = 2  # Need at least 2 for splitting
        
        if label_counts.min() < min_samples_per_stratum:
            print(f"Warning: Some strata have fewer than {min_samples_per_stratum} samples. Using simple random split.")
            stratification_labels = None
    else:
        stratification_labels = None
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test, temp_indices, test_indices = train_test_split(
        X, y, indices if indices is not None else np.arange(len(X)), 
        test_size=TEST_RATIO, 
        random_state=SEED,
        stratify=stratification_labels if stratification_labels is not None else None
    )
    
    # Update stratification for remaining data if needed
    if stratification_labels is not None:
        temp_stratification = pd.Series(stratification_labels).iloc[temp_indices].values
        
        # Check again for the temp split
        if pd.Series(temp_stratification).value_counts().min() < 2:
            print("Warning: Insufficient samples for stratification in train/val split. Using simple random split.")
            temp_stratification = None
    else:
        temp_stratification = None
    
    # Second split: separate train and validation from remaining data
    val_size_from_temp = VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO)
    X_train, X_val, y_train, y_val, train_indices, val_indices = train_test_split(
        X_temp, y_temp, temp_indices,
        test_size=val_size_from_temp,
        random_state=SEED,
        stratify=temp_stratification if temp_stratification is not None else None
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_indices, val_indices, test_indices

def _build_multidim_strata(df_sub: pd.DataFrame) -> np.ndarray:
    """Build combined stratification labels: price-quantile × moneyness × time buckets."""
    required = {'mid_price', 'spx_close', 'strike_price', 'days_to_maturity'}
    if not required.issubset(df_sub.columns):
        return pd.qcut(df_sub['mid_price'].values, q=5, labels=False, duplicates='drop')

    price_q = pd.qcut(df_sub['mid_price'].values, q=5, labels=False, duplicates='drop')
    mny = (df_sub['spx_close'].values / np.maximum(df_sub['strike_price'].values, 1e-12))
    t_years = np.maximum(df_sub['days_to_maturity'].values, 0) / 365.25

    mny_bucket = pd.cut(mny, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS)
    t_bucket = pd.cut(t_years, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS)

    combo = pd.Series(price_q).astype(str) + '_' + mny_bucket.astype(str) + '_' + t_bucket.astype(str)
    counts = combo.value_counts()
    rare = counts[counts < 10].index
    combo = combo.mask(combo.isin(rare), other='OTHER')

    if (combo == 'OTHER').mean() > 0.8:
        return price_q
    return combo.values

def set_seed(seed):
    """Set seed for reproducibility"""
    np.random.seed(seed)
    
def create_results_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if USE_TIMESTAMP:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(script_dir, RESULTS_FOLDER, f"ols_run_{timestamp}")
    else:
        results_dir = os.path.join(script_dir, RESULTS_FOLDER)
    
    try:
        os.makedirs(results_dir, exist_ok=True)
        print(f"Created results directory: {results_dir}")
    except Exception as e:
        print(f"Error creating results directory: {e}")
        raise
    return results_dir

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    
    # Convert date column if it exists for time-based splitting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"Initial dataset size: {len(df)}")
    
    # Apply percentage bid-ask spread filter FIRST
    if SPREAD_FILTER_ENABLED and {'best_bid','best_offer','mid_price'}.issubset(df.columns):
        initial_count = len(df)
        valid_mid = df['mid_price'] > 0
        valid_spread_order = df['best_offer'] >= df['best_bid']
        spread_pct = 100.0 * (df['best_offer'] - df['best_bid']) / df['mid_price']
        mask = valid_mid & valid_spread_order
        if MIN_SPREAD_PCT is not None:
            mask = mask & (spread_pct >= MIN_SPREAD_PCT)
        if MAX_SPREAD_PCT is not None:
            mask = mask & (spread_pct <= MAX_SPREAD_PCT)
        df = df[mask]
        removed = initial_count - len(df)
        print(f"Applied % spread filter [{MIN_SPREAD_PCT:.2f}%, {MAX_SPREAD_PCT:.2f}%] and ask>=bid: removed {removed:,} rows; remaining {len(df):,}")
    
    # Apply moneyness filtering
    if 'moneyness' in df.columns and (MONEYNESS_LOWER_BOUND is not None or MONEYNESS_UPPER_BOUND is not None):
        initial_count = len(df)
        if MONEYNESS_LOWER_BOUND is not None and MONEYNESS_UPPER_BOUND is not None:
            df = df[(df['moneyness'] >= MONEYNESS_LOWER_BOUND) & (df['moneyness'] <= MONEYNESS_UPPER_BOUND)]
            print(f"Applied moneyness filter ({MONEYNESS_LOWER_BOUND} ≤ S/K ≤ {MONEYNESS_UPPER_BOUND}): {len(df):,} rows (removed {initial_count - len(df):,})")
        elif MONEYNESS_LOWER_BOUND is not None:
            df = df[df['moneyness'] >= MONEYNESS_LOWER_BOUND]
            print(f"Applied moneyness filter (S/K ≥ {MONEYNESS_LOWER_BOUND}): {len(df):,} rows (removed {initial_count - len(df):,})")
        elif MONEYNESS_UPPER_BOUND is not None:
            df = df[df['moneyness'] <= MONEYNESS_UPPER_BOUND]
            print(f"Applied moneyness filter (S/K ≤ {MONEYNESS_UPPER_BOUND}): {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Apply time to maturity filtering
    if 'days_to_maturity' in df.columns and (MIN_DAYS_TO_MATURITY is not None or MAX_DAYS_TO_MATURITY is not None):
        initial_count = len(df)
        if MIN_DAYS_TO_MATURITY is not None and MAX_DAYS_TO_MATURITY is not None:
            df = df[(df['days_to_maturity'] >= MIN_DAYS_TO_MATURITY) & (df['days_to_maturity'] <= MAX_DAYS_TO_MATURITY)]
            print(f"Applied time to maturity filter ({MIN_DAYS_TO_MATURITY} ≤ days ≤ {MAX_DAYS_TO_MATURITY}): {len(df):,} rows (removed {initial_count - len(df):,})")
        elif MIN_DAYS_TO_MATURITY is not None:
            df = df[df['days_to_maturity'] >= MIN_DAYS_TO_MATURITY]
            print(f"Applied time to maturity filter (days ≥ {MIN_DAYS_TO_MATURITY}): {len(df):,} rows (removed {initial_count - len(df):,})")
        elif MAX_DAYS_TO_MATURITY is not None:
            df = df[df['days_to_maturity'] <= MAX_DAYS_TO_MATURITY]
            print(f"Applied time to maturity filter (days ≤ {MAX_DAYS_TO_MATURITY}): {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Apply quality filters
    initial_count = len(df)
    
    # Filter 1: Ask > Bid (valid bid-ask spread)
    if FILTER_VALID_SPREAD:  
        valid_spread = df['best_offer'] > df['best_bid']
        df = df[valid_spread]
        print(f"Removed {initial_count - len(df)} options with invalid bid-ask spread (ask <= bid)")
    
    # Filter 2: Mid price >= 0.05
    MIN_MEANINGFUL_PRICE = 0.05
    initial_count = len(df)
    df = df[df['mid_price'] >= MIN_MEANINGFUL_PRICE]
    print(f"Removed {initial_count - len(df)} options with mid price < ${MIN_MEANINGFUL_PRICE}")
    
    # Filter 3: Legacy spread ratio relative to mid
    if not SPREAD_FILTER_ENABLED and {'best_bid','best_offer','mid_price'}.issubset(df.columns):
        MAX_SPREAD_RATIO = 100
        initial_count = len(df)
        spread_ratio = (df['best_offer'] - df['best_bid']) / df['mid_price']
        df = df[spread_ratio <= MAX_SPREAD_RATIO]
        print(f"Removed {initial_count - len(df)} options with bid-ask spread ratio > {MAX_SPREAD_RATIO*100:.0f}% (legacy)")
    
    # Remove rows with missing values in key columns
    initial_count = len(df)
    DEFAULT_FEATURES = ['spx_close', 'strike_price', 'days_to_maturity', 'risk_free_rate', 'dividend_rate', 'historical_volatility']
    TARGET_COLUMN = 'mid_price'
    existing_required_cols = [col for col in DEFAULT_FEATURES + [TARGET_COLUMN] if col in df.columns]
    df = df.dropna(subset=existing_required_cols)
    print(f"After removing rows with missing values in key columns: {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Filter out invalid and non-finite values
    initial_count = len(df)
    volatility_col = 'historical_volatility' if 'historical_volatility' in df.columns else 'impl_volatility'
    
    is_finite_cols = (
        df['days_to_maturity'].replace([np.inf, -np.inf], np.nan).notna() &
        df['mid_price'].replace([np.inf, -np.inf], np.nan).notna() &
        df['strike_price'].replace([np.inf, -np.inf], np.nan).notna() &
        df['spx_close'].replace([np.inf, -np.inf], np.nan).notna() &
        df[volatility_col].replace([np.inf, -np.inf], np.nan).notna() &
        df['risk_free_rate'].replace([np.inf, -np.inf], np.nan).notna() &
        df['dividend_rate'].replace([np.inf, -np.inf], np.nan).notna()
    )

    valid_mask = (
        is_finite_cols &
        (df['days_to_maturity'] >= 0) &
        (df['mid_price'] > 0) &
        (df['strike_price'] > 0) &
        (df['spx_close'] > 0) &
        (df[volatility_col] >= 0)
    )
    
    # Cap volatility to avoid numerical issues
    MAX_REASONABLE_VOL = 5.0  # 500% annualized
    valid_mask = valid_mask & (df[volatility_col] <= MAX_REASONABLE_VOL)
    
    df = df[valid_mask]
    print(f"After filtering invalid values: {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Zero volume filtering
    if 'volume' in df.columns and ZERO_VOLUME_INCLUSION < 1.0:
        initial_count = len(df)
        zero_volume_mask = df['volume'] == 0
        non_zero_volume_df = df[~zero_volume_mask]
        zero_volume_df = df[zero_volume_mask]
        
        print(f"Zero volume options: {len(zero_volume_df):,} rows")
        print(f"Non-zero volume options: {len(non_zero_volume_df):,} rows")
        
        if ZERO_VOLUME_INCLUSION > 0.0 and len(zero_volume_df) > 0:
            n_zero_volume_to_keep = int(len(zero_volume_df) * ZERO_VOLUME_INCLUSION)
            if n_zero_volume_to_keep > 0:
                zero_volume_sample = zero_volume_df.sample(n=n_zero_volume_to_keep, random_state=SEED)
                df = pd.concat([non_zero_volume_df, zero_volume_sample], ignore_index=True)
                print(f"Kept {n_zero_volume_to_keep:,} zero-volume options ({ZERO_VOLUME_INCLUSION:.1%})")
            else:
                df = non_zero_volume_df
                print("Excluded all zero-volume options")
        else:
            df = non_zero_volume_df
            print("Excluded all zero-volume options")
        
        print(f"Applied zero volume filter (inclusion rate: {ZERO_VOLUME_INCLUSION:.1%}): {len(df):,} rows (removed {initial_count - len(df):,})")
        # --- Reorder rows to avoid volume-block bias before splitting ---
        # If we do a time-based percent split, ensure chronological ordering.
        # Otherwise, shuffle to remove any concatenation-induced blocks.
        if USE_TIME_BASED_SPLIT and USE_TIME_PERCENT_SPLIT and 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            print("Re-sorted dataset by date to support time-based percent split.")
        else:
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            print("Shuffled dataset rows to avoid ordering bias during random split.")

    # Sample if needed
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=SEED)
        print(f"Sampled {SAMPLE_SIZE} rows from {len(df)} available")
    
    print(f"Final dataset size: {len(df)}")
    
    # Prepare feature matrix with polynomial features
    target = 'mid_price'
    y = df[target].values
    
    available_base_features = [f for f in BASE_FEATURES if f in df.columns]
    available_additional_features = [f for f in ADDITIONAL_RAW_FEATURES if f in df.columns]
    
    if 'cp_flag' in df.columns:
        df['cp_flag_binary'] = (df['cp_flag'] == 'p').astype(int)
        available_additional_features.append('cp_flag_binary')
    
    X_base = df[available_base_features].values
    
    # Apply feature engineering
    if POLY_DEGREE > 1:
        print(f"\nApplying polynomial features (degree={POLY_DEGREE})...")
        poly = PolynomialFeatures(degree=POLY_DEGREE, include_bias=False, interaction_only=False)
        X_all_features = poly.fit_transform(X_base)
        all_feature_names_raw = poly.get_feature_names_out(available_base_features)
        
        # Filter out interaction terms if not wanted
        if not INCLUDE_INTERACTION_TERMS:
            keep_indices = []
            filtered_names = []
            
            for i, name in enumerate(all_feature_names_raw):
                if ' ' not in name or '^2' in name:
                    keep_indices.append(i)
                    filtered_names.append(name)
            
            X_all_features = X_all_features[:, keep_indices]
            all_feature_names_raw = filtered_names
        
        all_feature_names = [name.replace(' ', '_') for name in all_feature_names_raw]
        
        # Select features
        selected_indices = []
        selected_raw_features = []
        
        for feature in SELECTED_FEATURES:
            if feature in all_feature_names:
                selected_indices.append(list(all_feature_names).index(feature))
            elif feature in available_additional_features:
                selected_raw_features.append(feature)
        
        if selected_indices:
            X_poly_selected = X_all_features[:, selected_indices]
            poly_feature_names = [all_feature_names[i] for i in selected_indices]
        else:
            X_poly_selected = np.empty((X_all_features.shape[0], 0))
            poly_feature_names = []
        
        if selected_raw_features:
            X_raw_selected = df[selected_raw_features].values
        else:
            X_raw_selected = np.empty((df.shape[0], 0))
        
        # Combine polynomial and raw features
        if X_poly_selected.shape[1] > 0 and X_raw_selected.shape[1] > 0:
            X_selected = np.column_stack([X_poly_selected, X_raw_selected])
            selected_feature_names = poly_feature_names + selected_raw_features
        elif X_poly_selected.shape[1] > 0:
            X_selected = X_poly_selected
            selected_feature_names = poly_feature_names
        else:
            X_selected = X_raw_selected
            selected_feature_names = selected_raw_features
        
        return X_selected, y, df, selected_feature_names
    else:
        return df[available_base_features].values, y, df, available_base_features

def fit_econometric_ols(X_train, y_train, X_test, y_test, feature_names, results_dir):
    """Fit OLS using statsmodels for detailed statistical analysis"""
    print("\n=== Fitting Statsmodels OLS (Detailed Analysis) ===")
    
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    # Fit models
    model = sm.OLS(y_train, X_train_const).fit()
    model_robust = sm.OLS(y_train, X_train_const).fit(cov_type='HC3')
    
    # Predictions
    y_test_pred = model.predict(X_test_const)
    y_train_pred = model.predict(X_train_const)
    
    residuals = y_test - y_test_pred
    
    # Calculate metrics
    k = X_test.shape[1] + 1
    n_train = len(y_train)
    n_test = len(y_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / max(n_train - k, 1)
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / max(n_test - k, 1)
    
    dw_stat = durbin_watson(residuals)
    
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    mape = np.mean(np.abs((y_test - y_test_pred) / np.maximum(np.abs(y_test), 1e-8))) * 100
    
    detailed_metrics = {
        'mse': mse, 'mae': mae, 'mape': mape, 'MSE': mse, 'MAE': mae, 'RMSE': np.sqrt(mse),
        'r2': r2_test, 'adj_r2': adj_r2_test, 'R2': r2_test,
        'r2_test': r2_test, 'adj_r2_test': adj_r2_test,
        'r2_train': r2_train, 'adj_r2_train': adj_r2_train,
        'f_statistic': model.fvalue, 'f_pvalue': model.f_pvalue,
        'f_statistic_robust': model_robust.fvalue, 'f_pvalue_robust': model_robust.f_pvalue,
        'durbin_watson': dw_stat, 'n_features': len(feature_names),
        'n_train': len(X_train), 'n_test': len(X_test)
    }
    
    print(f"Test R²: {r2_test:.4f}")
    print(f"Test Adjusted R²: {adj_r2_test:.4f}")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
    print(f"F-statistic: {model.fvalue:.2f} (p-value: {model.f_pvalue:.2e})")
    print(f"Durbin-Watson: {dw_stat:.3f}")
    
    # Save summaries
    with open(os.path.join(results_dir, 'ols_detailed_summary.txt'), 'w') as f:
        f.write(str(model.summary()))
    with open(os.path.join(results_dir, 'ols_robust_summary.txt'), 'w') as f:
        f.write(str(model_robust.summary()))
    
    return model, detailed_metrics, y_test_pred, residuals

def perform_additional_econometric_tests(model, X_train_const, y_train, X_test_const, y_test, feature_names):
    """Perform additional econometric tests: VIF, Breusch-Pagan, Ramsey RESET"""
    print("\n=== Additional Econometric Tests ===")
    
    test_results = {}
    
    # VIF Test
    print("\n1. Variance Inflation Factor (VIF):")
    try:
        vif_data = []
        for i in range(1, X_train_const.shape[1]):
            try:
                vif_value = variance_inflation_factor(X_train_const, i)
                if np.isinf(vif_value) or vif_value > 1000:
                    vif_value = np.nan
            except:
                vif_value = np.nan
            feature_name = feature_names[i-1] if i-1 < len(feature_names) else f"Feature_{i-1}"
            vif_data.append({'Feature': feature_name, 'VIF': vif_value})

        vif_data.sort(key=lambda x: x['VIF'], reverse=True)
        
        high_vif_count = 0
        for item in vif_data[:10]:
            vif_status = "HIGH" if item['VIF'] > 10 else "OK"
            if item['VIF'] > 10:
                high_vif_count += 1
            print(f"   {item['Feature']:<25}: {item['VIF']:>8.2f} ({vif_status})")
        
        test_results['vif'] = {
            'data': vif_data, 'high_vif_count': high_vif_count,
            'max_vif': max(item['VIF'] for item in vif_data),
            'mean_vif': np.mean([item['VIF'] for item in vif_data])
        }
        
    except Exception as e:
        test_results['vif'] = {'error': str(e)}
    
    # Breusch-Pagan Test
    print("\n2. Breusch-Pagan Test:")
    try:
        residuals = model.resid
        bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = het_breuschpagan(residuals, X_train_const)
        
        test_results['breusch_pagan'] = {
            'lm_statistic': bp_lm, 'lm_pvalue': bp_lm_pvalue,
            'f_statistic': bp_fvalue, 'f_pvalue': bp_f_pvalue
        }
        
        bp_result = "REJECT" if bp_lm_pvalue < 0.05 else "FAIL TO REJECT"
        print(f"   LM Statistic: {bp_lm:.4f}, p-value: {bp_lm_pvalue:.4f}")
        print(f"   Result: {bp_result} null hypothesis of homoskedasticity")
        
    except Exception as e:
        test_results['breusch_pagan'] = {'error': str(e)}
    
    # Ramsey RESET Test
    print("\n3. Ramsey RESET Test:")
    try:
        reset_result = linear_reset(model, power=2, test_type='fitted')
        
        test_results['ramsey_reset'] = {
            'f_statistic': reset_result.statistic,
            'f_pvalue': reset_result.pvalue, 'power': 2
        }
        
        reset_result_status = "REJECT" if reset_result.pvalue < 0.05 else "FAIL TO REJECT"
        print(f"   F Statistic: {reset_result.statistic:.4f}, p-value: {reset_result.pvalue:.4f}")
        print(f"   Result: {reset_result_status} null hypothesis of correct functional form")
        
    except Exception as e:
        test_results['ramsey_reset'] = {'error': str(e)}
    
    return test_results

# VISUALIZATION FUNCTIONS

def create_binned_bar_chart(data, residuals, num_bins=50, colormap='viridis', 
                           xlabel='', ylabel='Average Absolute Residuals ($)', 
                           title='', ax=None):
    """Helper function to create binned bar charts"""
    if ax is None:
        ax = plt.gca()
    
    bins = np.linspace(data.min(), data.max(), num_bins)
    abs_residuals = np.abs(residuals)
    
    bin_avg_residuals = []
    bin_centers = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        bin_mask = (data >= bins[i]) & (data < bins[i+1])
        if bin_mask.sum() > 0:
            bin_avg_residuals.append(abs_residuals[bin_mask].mean())
            bin_centers.append((bins[i] + bins[i+1]) / 2)
            bin_counts.append(bin_mask.sum())
    
    if not bin_counts:
        return
    
    bin_width = (bins[1] - bins[0]) * 0.8
    norm = colors.Normalize(vmin=min(bin_counts), vmax=max(bin_counts))
    cmap = cm.get_cmap(colormap)
    bar_colors = [cmap(norm(count)) for count in bin_counts]
    
    bars = ax.bar(bin_centers, bin_avg_residuals, width=bin_width, 
                  color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Data Points Count')

def plot_residuals_histograms(y_true, y_pred, results_dir):
    """Create histograms for residuals analysis"""
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    pct_error = 100 * residuals / np.maximum(np.abs(y_true), 1e-8)
    abs_pct_error = np.abs(pct_error)
    
    plt.figure(figsize=(16, 12))
    
    # 1. Absolute Residuals Distribution
    plt.subplot(2, 2, 1)
    abs_residuals_95th = np.percentile(abs_residuals, 95)
    abs_residuals_filtered = abs_residuals[abs_residuals <= abs_residuals_95th]
    plt.hist(abs_residuals_filtered, bins=50, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
    plt.xlabel('Absolute Residuals ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Absolute Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    plt.subplot(2, 2, 2)
    residuals_5th = np.percentile(residuals, 5)
    residuals_95th = np.percentile(residuals, 95)
    residuals_filtered = residuals[(residuals >= residuals_5th) & (residuals <= residuals_95th)]
    plt.hist(residuals_filtered, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Percentage Error Distribution
    plt.subplot(2, 2, 3)
    pct_error_filtered = pct_error[(pct_error >= -100) & (pct_error <= 100)]
    plt.hist(pct_error_filtered, bins=50, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
    plt.xlabel('Percentage Error (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Percentage Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Absolute Percentage Error Distribution
    plt.subplot(2, 2, 4)
    abs_pct_error_filtered = abs_pct_error[abs_pct_error <= 100]
    plt.hist(abs_pct_error_filtered, bins=50, alpha=0.7, density=True, color='gold', edgecolor='black')
    plt.xlabel('Absolute Percentage Error (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Absolute Percentage Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    save_path = os.path.join(results_dir, 'residuals_histograms.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals histograms saved to {save_path}")

def plot_residuals_histograms_frequency(y_true, y_pred, results_dir):
    """Create histograms for residuals analysis with frequency on y-axis"""
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    pct_error = 100 * residuals / np.maximum(np.abs(y_true), 1e-8)
    abs_pct_error = np.abs(pct_error)
    
    plt.figure(figsize=(16, 12))
    
    # 1. Absolute Residuals Distribution
    plt.subplot(2, 2, 1)
    abs_residuals_95th = np.percentile(abs_residuals, 95)
    abs_residuals_filtered = abs_residuals[abs_residuals <= abs_residuals_95th]
    plt.hist(abs_residuals_filtered, bins=50, alpha=0.7, density=False, color='lightcoral', edgecolor='black')
    plt.xlabel('Absolute Residuals ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Absolute Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    plt.subplot(2, 2, 2)
    residuals_5th = np.percentile(residuals, 5)
    residuals_95th = np.percentile(residuals, 95)
    residuals_filtered = residuals[(residuals >= residuals_5th) & (residuals <= residuals_95th)]
    plt.hist(residuals_filtered, bins=50, alpha=0.7, density=False, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Percentage Error Distribution
    plt.subplot(2, 2, 3)
    pct_error_filtered = pct_error[(pct_error >= -100) & (pct_error <= 100)]
    plt.hist(pct_error_filtered, bins=50, alpha=0.7, density=False, color='lightgreen', edgecolor='black')
    plt.xlabel('Percentage Error (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Percentage Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Absolute Percentage Error Distribution
    plt.subplot(2, 2, 4)
    abs_pct_error_filtered = abs_pct_error[abs_pct_error <= 100]
    plt.hist(abs_pct_error_filtered, bins=50, alpha=0.7, density=False, color='gold', edgecolor='black')
    plt.xlabel('Absolute Percentage Error (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Absolute Percentage Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    save_path = os.path.join(results_dir, 'residuals_histograms_frequency.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals histograms (frequency) saved to {save_path}")

def plot_residuals_vs_features(y_true, y_pred, X_test, df_test, results_dir):
    """Create residuals vs features plots with bar charts"""
    residuals = y_pred - y_true
    
    plt.figure(figsize=(16, 12))
    
    # 1. Residuals vs Historical Volatility
    plt.subplot(2, 2, 1)
    if 'historical_volatility' in df_test.columns:
        hist_vol = df_test['historical_volatility'].values
        plt.scatter(hist_vol, residuals, alpha=0.3, s=1, color='blue')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Historical Volatility', fontsize=12)
        plt.ylabel('Residuals ($)', fontsize=12)
        plt.title('Residuals vs Historical Volatility', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Historical Volatility\nNot Available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Residuals vs Historical Volatility', fontsize=14, fontweight='bold')
    
    # 2. Residuals vs Fitted Values (Log Scale)
    plt.subplot(2, 2, 2)
    valid_mask = (y_pred > 0) & (residuals != 0)
    y_pred_valid = y_pred[valid_mask]
    residuals_valid = residuals[valid_mask]
    
    plt.scatter(y_pred_valid, residuals_valid, alpha=0.3, s=1, color='green')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values (Log Scale)', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs Fitted Values (Log)', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # 3. Average Absolute Residuals vs Moneyness
    plt.subplot(2, 2, 3)
    if 'moneyness' in df_test.columns:
        moneyness = df_test['moneyness'].values
        create_binned_bar_chart(moneyness, residuals, colormap='viridis',
                               xlabel='Moneyness (S/K)', title='Avg Abs Residuals vs Moneyness')
    
    # 4. Average Absolute Residuals vs Days to Maturity
    plt.subplot(2, 2, 4)
    if 'days_to_maturity' in df_test.columns:
        days_to_maturity = df_test['days_to_maturity'].values
        create_binned_bar_chart(days_to_maturity, residuals, colormap='plasma',
                               xlabel='Days to Maturity', title='Avg Abs Residuals vs Days to Maturity')
    
    plt.tight_layout(pad=3.0)
    save_path = os.path.join(results_dir, 'residuals_vs_features.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals vs features plots saved to {save_path}")

def create_mae_analysis(y_true, y_pred, df_test, results_dir):
    """Create MAE analysis by features"""
    print("Creating MAE analysis by features...")
    
    if len(df_test) == 0:
        print("   Warning: Empty dataset, skipping MAE analysis")
        return
    
    # Add price_error column to df_test for analysis
    df_analysis = df_test.copy()
    df_analysis['price_error'] = y_pred - y_true
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. MAE by Price Quantiles
    ax1 = axes[0, 0]
    price_quantiles = pd.qcut(y_true, q=5, labels=False, duplicates='drop')
    mae_by_quantile = []
    quantile_labels = []
    quantile_counts = []
    
    unique_quantiles = sorted(np.unique(price_quantiles[~pd.isna(price_quantiles)]))
    for q in unique_quantiles:
        mask = price_quantiles == q
        if mask.sum() > 0:
            subset_prices = y_true[mask]
            min_price = subset_prices.min()
            max_price = subset_prices.max()
            mae_by_quantile.append(np.mean(np.abs(df_analysis[mask]['price_error'])))
            quantile_labels.append(f'${min_price:.0f}-\n${max_price:.0f}')
            quantile_counts.append(mask.sum())
    
    bars1 = ax1.bar(quantile_labels, mae_by_quantile, color='lightblue', alpha=0.8, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars1, quantile_counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)
    ax1.set_title('Mean Absolute Error by Price Quantiles', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. MAE by Moneyness
    ax2 = axes[0, 1]
    if 'moneyness' in df_analysis.columns:
        df_analysis['moneyness_bucket'] = get_moneyness_buckets(df_analysis['moneyness'])
        moneyness_analysis = df_analysis.groupby('moneyness_bucket').agg({
            'price_error': lambda x: np.mean(np.abs(x)),
            'moneyness': 'count'
        }).reset_index()
        moneyness_analysis.columns = ['moneyness_bucket', 'mae', 'count']
        
        bars2 = ax2.bar(range(len(moneyness_analysis)), moneyness_analysis['mae'], 
                       color='orange', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars2, moneyness_analysis['count'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
        ax2.set_xticks(range(len(moneyness_analysis)))
        ax2.set_xticklabels(moneyness_analysis['moneyness_bucket'])
    else:
        ax2.text(0.5, 0.5, 'Moneyness data not available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Mean Absolute Error by Moneyness', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. MAE by Time to Expiration
    ax3 = axes[1, 0]
    df_analysis['tte_bucket'] = get_time_buckets(df_analysis['days_to_maturity'] / 365.25)
    tte_analysis = df_analysis.groupby('tte_bucket').agg({
        'price_error': lambda x: np.mean(np.abs(x)),
        'days_to_maturity': 'count'
    }).reset_index()
    tte_analysis.columns = ['tte_bucket', 'mae', 'count']
    
    bars3 = ax3.bar(range(len(tte_analysis)), tte_analysis['mae'], 
                   color='green', alpha=0.8, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars3, tte_analysis['count'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)
    ax3.set_xticks(range(len(tte_analysis)))
    ax3.set_xticklabels(tte_analysis['tte_bucket'])
    ax3.set_title('Mean Absolute Error by Time to Expiration', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. MAE by Historical Volatility
    ax4 = axes[1, 1]
    if 'historical_volatility' in df_analysis.columns:
        vol_quantiles = pd.qcut(df_analysis['historical_volatility'], q=5, labels=False, duplicates='drop')
        mae_by_vol = []
        vol_labels = []
        vol_counts = []
        
        unique_quantiles = sorted(vol_quantiles[~pd.isna(vol_quantiles)].unique())
        for q in unique_quantiles:
            mask = vol_quantiles == q
            if mask.sum() > 0:
                subset_vol = df_analysis[mask]['historical_volatility']
                min_vol = subset_vol.min()
                max_vol = subset_vol.max()
                mae_by_vol.append(np.mean(np.abs(df_analysis[mask]['price_error'])))
                vol_labels.append(f'{min_vol:.2f}-\n{max_vol:.2f}')
                vol_counts.append(mask.sum())
        
        bars4 = ax4.bar(vol_labels, mae_by_vol, color='purple', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars4, vol_counts)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'Historical volatility data not available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Mean Absolute Error by Historical Volatility', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. MAE by Volume (if available)
    ax5 = axes[2, 0]
    if 'volume' in df_analysis.columns:
        df_analysis['volume_bucket'] = get_volume_buckets(df_analysis['volume'])
        volume_analysis = df_analysis.groupby('volume_bucket').agg({
            'price_error': lambda x: np.mean(np.abs(x)),
            'volume': 'count'
        }).reset_index()
        volume_analysis.columns = ['volume_bucket', 'mae', 'count']
        
        bars5 = ax5.bar(range(len(volume_analysis)), volume_analysis['mae'], 
                       color='skyblue', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars5, volume_analysis['count'])):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
        ax5.set_xticks(range(len(volume_analysis)))
        ax5.set_xticklabels(volume_analysis['volume_bucket'], rotation=45, ha='right')
    else:
        ax5.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Mean Absolute Error by Volume', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. MAE by Equity Uncertainty
    ax6 = axes[2, 1]
    if 'equity_uncertainty' in df_analysis.columns:
        df_analysis['uncertainty_bucket'] = get_uncertainty_buckets(df_analysis['equity_uncertainty'])
        uncertainty_analysis = df_analysis.groupby('uncertainty_bucket').agg({
            'price_error': lambda x: np.mean(np.abs(x)),
            'equity_uncertainty': 'count'
        }).reset_index()
        uncertainty_analysis.columns = ['uncertainty_bucket', 'mae', 'count']
        
        create_bucket_bar_plot(ax6, uncertainty_analysis.set_index('uncertainty_bucket'), 
                              'mae', 'count', 'Mean Absolute Error by Equity Uncertainty', 
                              'Mean Absolute Error ($)', 'red', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'Equity uncertainty data not available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Mean Absolute Error by Equity Uncertainty', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(results_dir, 'mae_by_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("MAE by features analysis saved")

def create_mape_analysis(y_true, y_pred, df_test, results_dir):
    """Create MAPE analysis by features"""
    print("Creating MAPE analysis by features...")
    
    if len(df_test) == 0:
        print("   Warning: Empty dataset, skipping MAPE analysis")
        return
    
    # Calculate percentage errors
    pct_error = 100 * (y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)
    abs_pct_error = np.abs(pct_error)
    
    df_analysis = df_test.copy()
    df_analysis['abs_pct_error'] = abs_pct_error
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. MAPE by Price Quantiles
    ax1 = axes[0, 0]
    price_quantiles = pd.qcut(y_true, q=5, labels=False, duplicates='drop')
    mape_by_quantile = []
    quantile_labels = []
    quantile_counts = []
    
    unique_quantiles = sorted(np.unique(price_quantiles[~pd.isna(price_quantiles)]))
    for q in unique_quantiles:
        mask = price_quantiles == q
        if mask.sum() > 0:
            subset_prices = y_true[mask]
            min_price = subset_prices.min()
            max_price = subset_prices.max()
            mape_by_quantile.append(np.mean(df_analysis[mask]['abs_pct_error']))
            quantile_labels.append(f'${min_price:.0f}-\n${max_price:.0f}')
            quantile_counts.append(mask.sum())
    
    bars1 = ax1.bar(quantile_labels, mape_by_quantile, color='lightblue', alpha=0.8, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars1, quantile_counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)
    ax1.set_title('MAPE by Price Quantiles', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 2. MAPE by Moneyness
    ax2 = axes[0, 1]
    if 'moneyness' in df_analysis.columns:
        df_analysis['moneyness_bucket'] = get_moneyness_buckets(df_analysis['moneyness'])
        moneyness_analysis = df_analysis.groupby('moneyness_bucket').agg({
            'abs_pct_error': 'mean',
            'moneyness': 'count'
        }).reset_index()
        moneyness_analysis.columns = ['moneyness_bucket', 'mape', 'count']
        
        bars2 = ax2.bar(range(len(moneyness_analysis)), moneyness_analysis['mape'], 
                       color='orange', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars2, moneyness_analysis['count'])):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
        ax2.set_xticks(range(len(moneyness_analysis)))
        ax2.set_xticklabels(moneyness_analysis['moneyness_bucket'])
    else:
        ax2.text(0.5, 0.5, 'Moneyness data not available', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
    ax2.set_title('MAPE by Moneyness', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. MAPE by Time to Expiration
    ax3 = axes[1, 0]
    df_analysis['tte_bucket'] = get_time_buckets(df_analysis['days_to_maturity'] / 365.25)
    tte_analysis = df_analysis.groupby('tte_bucket').agg({
        'abs_pct_error': 'mean',
        'days_to_maturity': 'count'
    }).reset_index()
    tte_analysis.columns = ['tte_bucket', 'mape', 'count']
    
    bars3 = ax3.bar(range(len(tte_analysis)), tte_analysis['mape'], 
                   color='green', alpha=0.8, edgecolor='black')
    for i, (bar, count) in enumerate(zip(bars3, tte_analysis['count'])):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'n={count:,}', ha='center', va='bottom', fontsize=8)
    ax3.set_xticks(range(len(tte_analysis)))
    ax3.set_xticklabels(tte_analysis['tte_bucket'])
    ax3.set_title('MAPE by Time to Expiration', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. MAPE by Historical Volatility
    ax4 = axes[1, 1]
    if 'historical_volatility' in df_analysis.columns:
        vol_quantiles = pd.qcut(df_analysis['historical_volatility'], q=5, labels=False, duplicates='drop')
        mape_by_vol = []
        vol_labels = []
        vol_counts = []
        
        unique_quantiles = sorted(vol_quantiles[~pd.isna(vol_quantiles)].unique())
        for q in unique_quantiles:
            mask = vol_quantiles == q
            if mask.sum() > 0:
                subset_vol = df_analysis[mask]['historical_volatility']
                min_vol = subset_vol.min()
                max_vol = subset_vol.max()
                mape_by_vol.append(np.mean(df_analysis[mask]['abs_pct_error']))
                vol_labels.append(f'{min_vol:.2f}-\n{max_vol:.2f}')
                vol_counts.append(mask.sum())
        
        bars4 = ax4.bar(vol_labels, mape_by_vol, color='purple', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars4, vol_counts)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'Historical volatility data not available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
    ax4.set_title('MAPE by Historical Volatility', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # 5. MAPE by Volume (if available)
    ax5 = axes[2, 0]
    if 'volume' in df_analysis.columns:
        df_analysis['volume_bucket'] = get_volume_buckets(df_analysis['volume'])
        volume_analysis = df_analysis.groupby('volume_bucket').agg({
            'abs_pct_error': 'mean',
            'volume': 'count'
        }).reset_index()
        volume_analysis.columns = ['volume_bucket', 'mape', 'count']
        
        bars5 = ax5.bar(range(len(volume_analysis)), volume_analysis['mape'], 
                       color='skyblue', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars5, volume_analysis['count'])):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
        ax5.set_xticks(range(len(volume_analysis)))
        ax5.set_xticklabels(volume_analysis['volume_bucket'], rotation=45, ha='right')
    else:
        ax5.text(0.5, 0.5, 'Volume data not available', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
    ax5.set_title('MAPE by Volume', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # 6. MAPE by Equity Uncertainty
    ax6 = axes[2, 1]
    if 'equity_uncertainty' in df_analysis.columns:
        df_analysis['uncertainty_bucket'] = get_uncertainty_buckets(df_analysis['equity_uncertainty'])
        uncertainty_analysis = df_analysis.groupby('uncertainty_bucket').agg({
            'abs_pct_error': 'mean',
            'equity_uncertainty': 'count'
        }).reset_index()
        uncertainty_analysis.columns = ['uncertainty_bucket', 'mape', 'count']
        
        create_bucket_bar_plot(ax6, uncertainty_analysis.set_index('uncertainty_bucket'), 
                              'mape', 'count', 'MAPE by Equity Uncertainty', 
                              'Mean Absolute Percentage Error (%)', 'red', rotation=45)
    else:
        ax6.text(0.5, 0.5, 'Equity uncertainty data not available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('MAPE by Equity Uncertainty', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(results_dir, 'mape_by_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("MAPE by features analysis saved")

def plot_normalized_error_histogram_ols(y_true, y_pred, df_test, results_dir):
    """Plot (pred-mid)/(half_spread) histogram"""
    print("Generating normalized error histogram (bid-ask spread analysis)...")
    
    if 'best_bid' not in df_test.columns or 'best_offer' not in df_test.columns:
        print("   Warning: Missing bid-ask data for normalized error analysis")
        return
    
    # Calculate half spread
    bid_prices = df_test['best_bid'].values
    ask_prices = df_test['best_offer'].values
    half_spread = (ask_prices - bid_prices) / 2
    
    # Calculate normalized error
    error = y_pred - y_true
    normalized_error = error / np.maximum(half_spread, 0.01)
    
    # Filter extreme outliers
    filtered_error = normalized_error[(normalized_error >= -10) & (normalized_error <= 10)]
    
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(filtered_error, bins=100, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.axvline(x=np.mean(filtered_error), color='orange', linestyle='-', linewidth=2, 
                label=f'Mean: {np.mean(filtered_error):.3f}')
    plt.xlabel('(Predicted - Actual) / Half Spread')
    plt.ylabel('Density')
    plt.title('OLS: Normalized Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 2, 2)
    sorted_errors = np.sort(np.abs(filtered_error))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, color='green', linewidth=2)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 Half Spread')
    plt.axvline(x=2, color='orange', linestyle='--', alpha=0.7, label='2 Half Spreads')
    plt.xlabel('|Normalized Error|')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution of |Normalized Error|')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics text
    plt.subplot(2, 2, 3)
    plt.axis('off')
    stats_text = f"""OLS Normalized Error Statistics:

Mean: {np.mean(filtered_error):.4f}
Median: {np.median(filtered_error):.4f}
Std Dev: {np.std(filtered_error):.4f}

Absolute Error Stats:
Mean |Error|: {np.mean(np.abs(filtered_error)):.4f}
Median |Error|: {np.median(np.abs(filtered_error)):.4f}

Within 1 Half Spread: {np.mean(np.abs(filtered_error) <= 1)*100:.1f}%
Within 2 Half Spreads: {np.mean(np.abs(filtered_error) <= 2)*100:.1f}%
Within 0.5 Half Spread: {np.mean(np.abs(filtered_error) <= 0.5)*100:.1f}%

Sample Size: {len(filtered_error):,}
Outliers Removed: {len(normalized_error) - len(filtered_error):,}"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    # Box plot by price ranges
    plt.subplot(2, 2, 4)
    price_quartiles = pd.qcut(y_true, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    quartile_data = []
    quartile_labels = []
    
    non_outlier_mask = (normalized_error >= -10) & (normalized_error <= 10)
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = (price_quartiles == q) & non_outlier_mask
        if mask.sum() > 0:
            quartile_errors = normalized_error[mask]
            quartile_data.append(quartile_errors)
            quartile_labels.append(f'{q}\n(n={len(quartile_errors)})')
    
    if quartile_data:
        plt.boxplot(quartile_data, labels=quartile_labels)
        plt.ylabel('Normalized Error')
        plt.xlabel('Price Quartiles')
        plt.title('Normalized Error by Price Range')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'ols_normalized_error_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   OLS normalized error histogram saved to {save_path}")

def plot_correlation_heatmap(X_selected, selected_feature_names, y, df, results_dir):
    """Plot correlation heatmap of selected features"""
    print("Generating correlation heatmap...")
    
    # Create DataFrame with selected features and target
    df_corr = pd.DataFrame(X_selected, columns=selected_feature_names)
    df_corr['mid_price'] = y
    
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    
    # Create the heatmap
    plt.figure(figsize=(16, 12))
    
    # Plot heatmap with correlation numbers
    sns.heatmap(corr_matrix, 
                annot=True,  # Show correlation numbers
                cmap='RdBu_r',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    
    plt.title('Correlation Matrix of Selected Features', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    heatmap_path = os.path.join(results_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Correlation heatmap saved to {heatmap_path}")
    
    # Also create a second heatmap showing all raw features
    plt.figure(figsize=(16, 12))
    
    raw_features = [
        'strike_price', 'volume', 'mid_price', 'days_to_maturity',
        'historical_volatility', 'hist_vol_10d', 'hist_vol_30d', 'hist_vol_90d',
        'risk_free_rate', 'dividend_rate', 'spx_open', 'spx_high', 'spx_low', 'spx_close',
        'moneyness'
    ]
    
    if 'cp_flag_binary' in df.columns:
        raw_features.append('cp_flag_binary')
    
    available_raw_features = [f for f in raw_features if f in df.columns]
    
    if len(available_raw_features) > 1:
        df_raw_corr = df[available_raw_features]
        raw_corr_matrix = df_raw_corr.corr()
        
        sns.heatmap(raw_corr_matrix,
                    annot=True,
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={'shrink': 0.8},
                    annot_kws={'size': 7})
        
        plt.title('All Raw Features Correlation Matrix', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        simplified_heatmap_path = os.path.join(results_dir, 'correlation_heatmap_all_features.png')
        plt.savefig(simplified_heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"All features correlation heatmap saved to {simplified_heatmap_path}")
    
    return corr_matrix

def plot_regression_diagnostics(y_true, y_pred, residuals, results_dir, df_test=None):
    """Create comprehensive regression diagnostic plots"""
    plt.figure(figsize=(15, 12))
    
    # 1. True vs Predicted
    plt.subplot(2, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.3, s=1)
    min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals vs Fitted Values
    plt.subplot(2, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.3, s=1, color='green')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values (Predicted Price $)')
    plt.ylabel('Residuals ($)')
    plt.title('Residuals vs Fitted Values')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals Distribution
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', density=True)
    # Overlay normal distribution
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residuals Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Q-Q Plot
    plt.subplot(2, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Test)')
    plt.grid(True, alpha=0.3)
    
    # 5. Scale-Location Plot
    plt.subplot(2, 3, 5)
    sqrt_abs_residuals = np.sqrt(np.abs(residuals))
    plt.scatter(y_pred, sqrt_abs_residuals, alpha=0.3, s=1)
    plt.xlabel('Fitted Values')
    plt.ylabel('√|Residuals|')
    plt.title('Scale-Location Plot')
    plt.grid(True, alpha=0.3)
    
    # 6. Residuals vs Order
    plt.subplot(2, 3, 6)
    plt.plot(residuals, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Observation Order')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Order')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'regression_diagnostics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Regression diagnostics saved to {save_path}")

def plot_feature_importance(feature_names, coefficients, results_dir):
    """Plot feature importance based on coefficient magnitudes"""
    plt.figure(figsize=(12, 8))
    
    abs_coefs = np.abs(coefficients)
    sorted_idx = np.argsort(abs_coefs)[::-1]
    
    n_features = min(20, len(feature_names))
    top_idx = sorted_idx[:n_features]
    
    plt.subplot(1, 2, 1)
    colors = ['red' if coef < 0 else 'blue' for coef in coefficients[top_idx]]
    plt.barh(range(n_features), coefficients[top_idx], color=colors, alpha=0.7)
    plt.yticks(range(n_features), [feature_names[i] for i in top_idx])
    plt.xlabel('Coefficient Value')
    plt.title('Feature Coefficients (Top 20)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.barh(range(n_features), abs_coefs[top_idx], color='green', alpha=0.7)
    plt.yticks(range(n_features), [feature_names[i] for i in top_idx])
    plt.xlabel('|Coefficient Value|')
    plt.title('Feature Importance (Top 20)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance plot saved to {save_path}")

def plot_performance_summary(metrics, results_dir, y_test=None, y_test_pred=None, residuals=None):
    """Create a performance summary visualization"""
    print("Generating performance summary...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('OLS Regression - Performance Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Performance Metrics Bar Chart
    metrics_names = ['R²', 'MSE', 'MAE', 'RMSE', 'MAPE']
    metrics_values = [
        metrics.get('R2', metrics.get('r2', 0)),
        metrics.get('MSE', metrics.get('mse', 0)),
        metrics.get('MAE', metrics.get('mae', 0)),
        metrics.get('RMSE', np.sqrt(metrics.get('mse', 0))),
        metrics.get('mape', 0)
    ]
    
    colors = ['skyblue', 'lightcoral', 'gold', 'lightgreen', 'plum']
    bars = axes[0, 0].bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Performance Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        if value < 1:
            label = f'{value:.4f}'
        elif value < 100:
            label = f'{value:.2f}'
        else:
            label = f'{value:.0f}'
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Model Information
    axes[0, 1].axis('off')
    info_text = f"""OLS Regression Model Summary:

Performance Metrics:
• R²: {metrics.get('R2', metrics.get('r2', 0)):.6f}
• MSE: {metrics.get('MSE', metrics.get('mse', 0)):.4f}
• RMSE: {metrics.get('RMSE', np.sqrt(metrics.get('mse', 0))):.4f}
• MAE: {metrics.get('MAE', metrics.get('mae', 0)):.4f}

Model Configuration:
• Polynomial Degree: {POLY_DEGREE}
• Base Features: {len(BASE_FEATURES)}
• Data Split: Time-based ({TIME_SPLIT_FRACTION:.1%} train)
• Zero Volume Inclusion: {ZERO_VOLUME_INCLUSION:.1%}
• Moneyness Range: {MONEYNESS_LOWER_BOUND}-{MONEYNESS_UPPER_BOUND}

"""
    
    axes[0, 1].text(0.05, 0.95, info_text, transform=axes[0, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))
    
    # Plot 3: Predicted vs Actual
    if y_test is not None and y_test_pred is not None:
        axes[1, 0].scatter(y_test, y_test_pred, alpha=0.5, s=1, color='blue')
        
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        range_padding = (max_val - min_val) * 0.05
        plot_min = min_val - range_padding
        plot_max = max_val + range_padding
        
        axes[1, 0].plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Perfect Prediction')
        axes[1, 0].set_xlim(plot_min, plot_max)
        axes[1, 0].set_ylim(plot_min, plot_max)
        axes[1, 0].set_xlabel('Actual Price')
        axes[1, 0].set_ylabel('Predicted Price')
        axes[1, 0].set_title(f'OLS: Predicted vs Actual\nR² = {metrics.get("R2", metrics.get("r2", 0)):.4f}')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Predicted vs Actual\n(Data not provided)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('OLS: Predicted vs Actual')
    
    # Plot 4: Residuals vs Predicted
    if y_test_pred is not None and residuals is not None:
        axes[1, 1].scatter(y_test_pred, residuals, alpha=0.5, s=1, color='orange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        axes[1, 1].set_xlabel('Predicted Price')
        axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
        axes[1, 1].set_title('OLS: Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Residuals vs Predicted\n(Data not provided)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('OLS: Residuals vs Predicted')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance summary saved to {save_path}")

def generate_comprehensive_ols_report(detailed_metrics, test_results, y_test, y_test_pred, residuals, 
                                     selected_feature_names, X_selected, df, df_test, results_dir):
    """Generate comprehensive OLS analysis report"""
    
    print("Generating comprehensive OLS analysis report...")
    
    # Calculate additional metrics
    rmse = np.sqrt(detailed_metrics['mse'])
    
    # Statistical tests on residuals
    if len(residuals) > 5000:
        sample_residuals = np.random.choice(residuals, 5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
        normality_note = " (tested on 5000 random sample)"
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        normality_note = ""
    
    # Jarque-Bera test for normality
    jb_stat, jb_p = stats.jarque_bera(residuals)
    
    # Calculate bid-ask spread accuracy
    if df_test is not None and 'best_bid' in df_test.columns and 'best_offer' in df_test.columns:
        bid_prices = df_test['best_bid'].values
        ask_prices = df_test['best_offer'].values
        in_spread = (y_test_pred >= bid_prices) & (y_test_pred <= ask_prices)
        spread_accuracy = float(np.mean(in_spread) * 100)
    else:
        spread_accuracy = None
    
    # Autocorrelation analysis
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
    else:
        autocorr = np.nan
    
    # Option type breakdown
    call_data = df[df['cp_flag'] == 'C'] if 'cp_flag' in df.columns else pd.DataFrame()
    put_data = df[df['cp_flag'] == 'P'] if 'cp_flag' in df.columns else pd.DataFrame()
    
    report_content = f"""
OLS REGRESSION - COMPREHENSIVE ANALYSIS REPORT
{'=' * 60}

Configuration:
- Base Features: {len(BASE_FEATURES)}
- Selected Features: {SELECTED_FEATURES}
- Total Features: {X_selected.shape[1]}
- Sample Size Limit: {SAMPLE_SIZE if SAMPLE_SIZE else 'None (all data)'}
- Zero Volume Inclusion Rate: {ZERO_VOLUME_INCLUSION:.1%}
- Train Ratio: {TRAIN_RATIO}
- Seed: {SEED}

Data Filtering Applied:
- Moneyness Filter: {MONEYNESS_LOWER_BOUND} ≤ S/K ≤ {MONEYNESS_UPPER_BOUND}
- Time to Maturity Filter: {MIN_DAYS_TO_MATURITY} ≤ days ≤ {MAX_DAYS_TO_MATURITY}

Final Dataset:
- Total Options: {len(df):,}
- Calls: {len(call_data):,} ({len(call_data)/len(df)*100:.1f}% if len(df) > 0 else 0)
- Puts: {len(put_data):,} ({len(put_data)/len(df)*100:.1f}% if len(df) > 0 else 0)
- Training samples: {detailed_metrics['n_train']:,}
- Test samples: {detailed_metrics['n_test']:,}

MODEL PERFORMANCE METRICS (Test unless specified):
- Mean Squared Error (MSE): {detailed_metrics['mse']:.4f}
- Mean Absolute Error (MAE): {detailed_metrics['mae']:.4f}
- Root Mean Square Error (RMSE): {rmse:.4f}
- Mean Absolute Percentage Error (MAPE): {detailed_metrics['mape']:.2f}%
- Test R-squared: {detailed_metrics['r2_test']:.4f}
- Test Adjusted R-squared: {detailed_metrics['adj_r2_test']:.4f}
- Train R-squared: {detailed_metrics['r2_train']:.4f}
- Train Adjusted R-squared: {detailed_metrics['adj_r2_train']:.4f}
- Bid-Ask Spread Accuracy (Test): {('N/A' if spread_accuracy is None else f"{spread_accuracy:.2f}%")}

RESIDUAL ANALYSIS:
- Mean Residual: {np.mean(residuals):.4f}
- Std Residual: {np.std(residuals):.4f}
- Skewness: {stats.skew(residuals):.4f}
- Kurtosis: {stats.kurtosis(residuals):.4f}

STATISTICAL TESTS:
- Shapiro-Wilk Normality Test{normality_note}:
  * Statistic: {shapiro_stat:.4f}
  * P-value: {shapiro_p:.2e}
  * Interpretation: {'Residuals appear normal' if shapiro_p > 0.05 else 'Residuals deviate from normality'}

- Jarque-Bera Normality Test:
  * Statistic: {jb_stat:.4f}
  * P-value: {jb_p:.2e}
  * Interpretation: {'Residuals appear normal' if jb_p > 0.05 else 'Residuals deviate from normality'}

- Autocorrelation Analysis:
  * Lag-1 Autocorrelation: {"N/A" if np.isnan(autocorr) else f"{autocorr:.4f}"}
  * Durbin-Watson Statistic: {detailed_metrics['durbin_watson']:.4f}
  * Interpretation: {"No significant autocorrelation" if not np.isnan(autocorr) and abs(autocorr) < 0.1 else "Potential autocorrelation detected"}

- F-Test (Model Significance):
  * F-statistic (Standard): {detailed_metrics['f_statistic']:.2f}
  * P-value (Standard): {detailed_metrics['f_pvalue']:.2e}
  * F-statistic (Robust): {detailed_metrics['f_statistic_robust']:.2f}
  * P-value (Robust): {detailed_metrics['f_pvalue_robust']:.2e}

ADDITIONAL ECONOMETRIC TESTS:
"""
    
    # Add test results
    if 'vif' in test_results and 'error' not in test_results['vif']:
        vif_data = test_results['vif']
        report_content += f"""- VIF Test (Multicollinearity):
  * Features with VIF > 10: {vif_data['high_vif_count']}
  * Maximum VIF: {vif_data['max_vif']:.2f}
  * Mean VIF: {vif_data['mean_vif']:.2f}
  * Interpretation: {'Multicollinearity detected' if vif_data['high_vif_count'] > 0 else 'No significant multicollinearity'}

"""
    
    if 'breusch_pagan' in test_results and 'error' not in test_results['breusch_pagan']:
        bp_data = test_results['breusch_pagan']
        bp_result = "Heteroskedasticity detected" if bp_data['lm_pvalue'] < 0.05 else "Homoskedasticity"
        report_content += f"""- Breusch-Pagan Test (Heteroskedasticity):
  * LM Statistic: {bp_data['lm_statistic']:.4f}
  * P-value: {bp_data['lm_pvalue']:.4f}
  * Interpretation: {bp_result}

"""
    
    if 'ramsey_reset' in test_results and 'error' not in test_results['ramsey_reset']:
        reset_data = test_results['ramsey_reset']
        reset_result = "Functional form misspecification" if reset_data['f_pvalue'] < 0.05 else "Correct functional form"
        report_content += f"""- Ramsey RESET Test (Functional Form):
  * F Statistic: {reset_data['f_statistic']:.4f}
  * P-value: {reset_data['f_pvalue']:.4f}
  * Interpretation: {reset_result}

"""
    
    # Add error statistics
    pct_error = (residuals / y_test) * 100
    abs_pct_error = np.abs(pct_error)
    
    report_content += f"""PREDICTION ERROR STATISTICS (Actual - Predicted):
count    {len(residuals):.0f}
mean     {np.mean(residuals):.6f}
std      {np.std(residuals):.6f}
min      {np.min(residuals):.6f}
25%      {np.percentile(residuals, 25):.6f}
50%      {np.percentile(residuals, 50):.6f}
75%      {np.percentile(residuals, 75):.6f}
max      {np.max(residuals):.6f}

PERCENTAGE ERROR STATISTICS:
count    {len(pct_error):.0f}
mean     {np.mean(pct_error):.6f}
std      {np.std(pct_error):.6f}
min      {np.min(pct_error):.6f}
25%      {np.percentile(pct_error, 25):.6f}
50%      {np.percentile(pct_error, 50):.6f}
75%      {np.percentile(pct_error, 75):.6f}
max      {np.max(pct_error):.6f}

FEATURE ANALYSIS:
- Selected Features: {', '.join(selected_feature_names)}
- Number of Features: {len(selected_feature_names)}

FILES GENERATED:
- ols_training_summary.txt (this comprehensive report)
- performance_summary.png (model performance visualization)
- regression_diagnostics.png (residual analysis)
- correlation_heatmap.png (feature correlations)
- ols_detailed_summary.txt (statsmodels output)
- ols_robust_summary.txt (robust standard errors)

"""
    
    # Add bucket analysis to the report
    bucket_results = perform_bucket_analysis(df_test, results_dir)
    
    def format_analysis_table(analysis_df, title):
        if analysis_df is None or analysis_df.empty:
            return f"{title}:\nNo data available.\n\n"
        
        table_str = f"{title}:\n\n"
        table_str += f"{'':25} {'Count':>10} {'MAE':>10} {'MAPE':>10} {'Median':>10} {'SD':>10}\n"
        
        for idx, row in analysis_df.iterrows():
            bucket_name = str(idx)
            count = int(row['Count'])
            mae = row['MAE']
            mape = row['MAPE']
            median = row['Median']
            sd = row['SD']
            
            table_str += f"{bucket_name:<25} {count:>10} {mae:>10.4f} {mape:>10.4f} {median:>10.4f} {sd:>10.4f}\n"
        
        table_str += "\n"
        return table_str
    
    # Add bucket analyses to report
    report_content += "\nBUCKET ANALYSIS:\n"
    report_content += "=" * 50 + "\n\n"
    
    for analysis_name, analysis_df in bucket_results.items():
        title_map = {
            'moneyness': 'MONEYNESS ANALYSIS',
            'time_to_expiration': 'Time to Expiration Analysis',
            'volume': 'Volume Bucket Analysis',
            'historical_volatility': 'Historical Volatility Bucket Analysis',
            'price_range': 'Price Range Analysis',
            'equity_uncertainty': 'Equity Uncertainty Analysis'
        }
        report_content += format_analysis_table(analysis_df, title_map.get(analysis_name, analysis_name.title()))
    
    report_content += f"""
INTERPRETATION:
{'=' * 20}
This OLS regression analysis provides theoretical baseline performance for comparison with other models.
The comprehensive statistical tests help validate model assumptions and identify potential issues.
"""
    
    # Save comprehensive report
    summary_path = os.path.join(results_dir, 'ols_training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(report_content)
    
    print(f"   Comprehensive analysis report saved to {summary_path}")
    return summary_path

# MAIN TRAINING FUNCTION

def train_ols_regression():
    """Main function to train and evaluate OLS regression models"""
    
    print("="*60)
    print("STARTING OLS REGRESSION ANALYSIS")
    print("="*60)
    
    setup_plot_style()
    set_seed(SEED)
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"Results will be saved to: {results_dir}")
    
    # Load and preprocess data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(script_dir), "final_options_dataset.csv")
    
    X_selected, y, df, selected_feature_names = load_dataset(csv_path)
    
    print(f"\nDataset loaded and preprocessed:")
    print(f"- Selected features: {len(selected_feature_names)}")
    print(f"- Final samples: {len(X_selected):,}")
    print(f"- Feature names: {selected_feature_names}")
    
    # Generate correlation heatmap
    plot_correlation_heatmap(X_selected, selected_feature_names, y, df, results_dir)
    
    df_view = df.reset_index(drop=True)
    n_samples = X_selected.shape[0]
    all_idx = np.arange(n_samples)
    
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = create_data_splits(
        X_selected, y, df_view, all_idx
    )

    print(f"\nData split:")
    print(f"Train: {len(train_idx):,} samples ({TRAIN_RATIO*100:.1f}%)")
    print(f"Val:   {len(val_idx):,} samples ({VALIDATION_RATIO*100:.1f}%)")
    print(f"Test:  {len(test_idx):,} samples ({TEST_RATIO*100:.1f}%)")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Features scaled using StandardScaler")
    
    # Fit econometric OLS model
    model, detailed_metrics, y_test_pred, residuals = fit_econometric_ols(
        X_train_scaled, y_train, X_test_scaled, y_test, selected_feature_names, results_dir
    )
    
    # Perform additional econometric tests
    X_train_const = sm.add_constant(X_train_scaled)
    X_test_const = sm.add_constant(X_test_scaled)
    test_results = perform_additional_econometric_tests(
        model, X_train_const, y_train, X_test_const, y_test, selected_feature_names
    )
    
    # Create test dataframe using exact test indices
    df_test = df.iloc[test_idx].copy()
    df_test['ols_pred'] = y_test_pred
    df_test['price_error'] = df_test['mid_price'] - df_test['ols_pred']
    df_test['abs_price_error'] = df_test['price_error'].abs()
    denom = np.maximum(np.abs(df_test['mid_price']), 1e-8)
    df_test['pct_error'] = 100.0 * df_test['price_error'] / denom
    df_test['abs_pct_error'] = df_test['pct_error'].abs()
    pred_csv_path = os.path.join(results_dir, 'options_with_ols_predictions.csv')
    df_test.to_csv(pred_csv_path, index=False)
    print(f"Enhanced OLS test subset saved to: {pred_csv_path}")
    print(f"   Columns added: ols_pred, price_error, abs_price_error, pct_error, abs_pct_error")
    
    # Generate visualizations
    plot_performance_summary(detailed_metrics, results_dir, y_test, y_test_pred, residuals)
    plot_regression_diagnostics(y_test, y_test_pred, residuals, results_dir, df_test)
    plot_feature_importance(selected_feature_names, model.params[1:], results_dir)  # Skip intercept
    
    # Additional plots
    plot_residuals_histograms(y_test, y_test_pred, results_dir)
    plot_residuals_histograms_frequency(y_test, y_test_pred, results_dir)
    plot_residuals_vs_features(y_test, y_test_pred, X_test, df_test, results_dir)
    create_mae_analysis(y_test, y_test_pred, df_test, results_dir)
    create_mape_analysis(y_test, y_test_pred, df_test, results_dir)
    plot_normalized_error_histogram_ols(y_test, y_test_pred, df_test, results_dir)
    
    # Generate comprehensive report
    summary_path = generate_comprehensive_ols_report(detailed_metrics, test_results, y_test, y_test_pred, residuals, 
                                                   selected_feature_names, X_selected, df, df_test, results_dir)
    
    print(f"\n=== Analysis Complete ===")
    print(f"All results saved to: {results_dir}")
    print(f"Analysis summary saved to: {summary_path}")

    print(f"\n" + "="*60)
    print("OLS REGRESSION ANALYSIS COMPLETE")
    print("="*60)
    print(f"All results saved to: {results_dir}")

if __name__ == "__main__":
    train_ols_regression()
