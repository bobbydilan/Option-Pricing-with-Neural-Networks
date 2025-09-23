"""Black-Scholes Option Pricing Analysis"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
plt.ioff()

ANALYSIS_MODE = 'historical'  # Options: 'historical' or 'implied'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BLACK_SCHOLES_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(PROJECT_ROOT, "final_options_dataset.csv")

SEED = 42
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
USE_TIME_BASED_SPLIT = True
TIME_SPLIT_DATE = '2019-01-01'
USE_TIME_PERCENT_SPLIT = True
TIME_SPLIT_FRACTION = 0.7
VAL_TEST_SPLIT = VALIDATION_RATIO / (1 - TRAIN_RATIO)
USE_TIMESTAMPED_OUTPUT = True

USE_MULTI_DIM_STRATIFICATION = False
SAMPLE_SIZE = None
ZERO_VOLUME_INCLUSION_RATE = 0.5
MONEYNESS_MIN = 0.5
MONEYNESS_MAX = 1.5
DAYS_TO_MATURITY_MIN = 0
DAYS_TO_MATURITY_MAX = 750
FILTER_VALID_SPREAD = False
SPREAD_FILTER_ENABLED = False
MIN_SPREAD_PCT = None
MAX_SPREAD_PCT = None
MIN_CLOSING_PRICE = None
MIN_MID_PRICE = None

MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
MONEYNESS_BIN_LABELS = ['OTM\n(<0.9)', 'ATM\n(0.9-1.1)', 'ITM\n(>1.1)']
TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]
TIME_BIN_LABELS = ['≤1M\n(≤30d)', '1-3M\n(31-91d)', '3-6M\n(92-182d)', '6-9M\n(183-274d)', '9-12M\n(275-365d)', '>12M\n(>365d)']
VOLUME_BINS_EDGES = [0, 10, 50, 200, 1000, np.inf]
VOLUME_BIN_LABELS = ['Very Low\n(0-10)', 'Low\n(11-50)', 'Medium\n(51-200)', 'High\n(201-1000)', 'Very High\n(>1000)']
UNCERTAINTY_BINS_EDGES = [0, 50, 100, 150, np.inf]
UNCERTAINTY_BIN_LABELS = ['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)']
HV_BINS_EDGES = [0, 0.15, 0.25, 0.35, 0.5, np.inf]
HV_BIN_LABELS = ['Very Low\n(<15%)', 'Low\n(15-25%)', 'Medium\n(25-35%)', 'High\n(35-50%)', 'Very High\n(>50%)']
PRICE_BINS_EDGES = [0, 10, 50, 100, 200, np.inf]
PRICE_BIN_LABELS = ['Very Low\n(<$10)', 'Low\n($10-50)', 'Medium\n($50-100)', 'High\n($100-200)', 'Very High\n(>$200)']

def setup_plot_style():
    """Set consistent plot style across scripts."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5
    })

def get_moneyness_buckets(moneyness_series: pd.Series):
    return pd.cut(moneyness_series, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)

def get_time_buckets(time_years_series: pd.Series):
    return pd.cut(time_years_series, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)

def get_volume_buckets(volume_series: pd.Series):
    volume_bins = [-0.1, 0, 100, 1000, np.inf]
    volume_labels = ['0', '0-100', '100-1000', '1000+']
    return pd.cut(volume_series, bins=volume_bins, labels=volume_labels, include_lowest=True)

def get_hv_buckets(hv_series: pd.Series):
    return pd.qcut(hv_series, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

def get_price_buckets(price_series: pd.Series):
    return pd.qcut(price_series, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

def get_uncertainty_buckets(uncertainty_series: pd.Series):
    return pd.cut(uncertainty_series, bins=UNCERTAINTY_BINS_EDGES, labels=UNCERTAINTY_BIN_LABELS, include_lowest=True)

# Mode-specific configurations
MODES = {
    'historical': {
        'volatility_col': 'historical_volatility',
        'output_dir': 'results',
        'output_file': 'options_with_black_scholes_historical.csv',
        'description': 'Theoretical Black-Scholes using historical volatility',
        'validation_type': 'Model validation against market prices'
    },
    'implied': {
        'volatility_col': 'impl_volatility',
        'output_dir': 'circular_check_results',
        'output_file': 'options_with_black_scholes_implied.csv',
        'description': 'Circular validation using implied volatility',
        'validation_type': 'Data consistency check (circular by construction)'
    }
}

# Stratification helper (mirror of MLP2)
def _build_multidim_strata(df_sub: pd.DataFrame) -> np.ndarray:
    """
    Build combined stratification labels: price-quantile × moneyness × time buckets.
    Mirrors MLP2 logic, including rare-class collapsing and fallback behavior.
    Returns an array of labels suitable for sklearn's stratify= argument.
    """
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

    return np.asarray(combo.values)

# CORE BLACK-SCHOLES FUNCTIONS

def black_scholes_price_only(S, K, T, r, sigma, option_type_array, q=0):
    """
    Simplified vectorized Black-Scholes calculation for prices only.
    Handles both calls and puts, with dividend yield support.
    """
    # Convert inputs to numpy arrays
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    # Edge-case guards
    eps = 1e-12
    # Replace non-positive underlyings/strikes with NaN to drop from valid calculations
    S = np.where(S > 0, S, np.nan)
    K = np.where(K > 0, K, np.nan)
    # Ensure non-negative time to maturity
    T = np.maximum(T, 0.0)
    # Replace non-finite rates/vols/dividends with NaN
    r = np.where(np.isfinite(r), r, np.nan)
    q = np.where(np.isfinite(q), q, np.nan)
    sigma = np.where(np.isfinite(sigma), sigma, np.nan)
    
    # Handle option type conversion
    if isinstance(option_type_array, str):
        option_type_array = [option_type_array]
    option_type_array = np.asarray(option_type_array)
    
    # Initialize output arrays
    prices = np.zeros_like(S)
    
    # Handle expired options (T <= 0)
    expired_mask = T <= 0
    # Valid = positive time and finite inputs
    valid_mask = (T > 0) & np.isfinite(S) & np.isfinite(K) & np.isfinite(r) & np.isfinite(q) & np.isfinite(sigma)
    
    # For expired options, use intrinsic value
    if np.any(expired_mask):
        call_mask = option_type_array == 'C'
        put_mask = option_type_array == 'P'
        
        # Intrinsic value for calls: max(S - K, 0)
        prices[expired_mask & call_mask] = np.maximum(S[expired_mask & call_mask] - K[expired_mask & call_mask], 0)
        # Intrinsic value for puts: max(K - S, 0)
        prices[expired_mask & put_mask] = np.maximum(K[expired_mask & put_mask] - S[expired_mask & put_mask], 0)
    
    # For valid options (T > 0), calculate Black-Scholes price
    if np.any(valid_mask):
        S_valid = S[valid_mask]
        K_valid = K[valid_mask]
        T_valid = T[valid_mask]
        r_valid = r[valid_mask]
        sigma_valid = sigma[valid_mask]
        q_valid = q[valid_mask]
        option_type_valid = option_type_array[valid_mask]

        # Split by volatility regime
        sigma_pos_mask = sigma_valid > 0
        sigma_zero_mask = ~sigma_pos_mask

        # Branch 1: Near-zero volatility -> forward intrinsic value
        if np.any(sigma_zero_mask):
            exp_neg_qT_zero = np.exp(-q_valid[sigma_zero_mask] * T_valid[sigma_zero_mask])
            exp_neg_rT_zero = np.exp(-r_valid[sigma_zero_mask] * T_valid[sigma_zero_mask])
            call_mask_zero = option_type_valid[sigma_zero_mask] == 'C'
            put_mask_zero = option_type_valid[sigma_zero_mask] == 'P'

            # Forward-discounted intrinsic
            call_val_zero = np.maximum(S_valid[sigma_zero_mask] * exp_neg_qT_zero - K_valid[sigma_zero_mask] * exp_neg_rT_zero, 0.0)
            put_val_zero = np.maximum(K_valid[sigma_zero_mask] * exp_neg_rT_zero - S_valid[sigma_zero_mask] * exp_neg_qT_zero, 0.0)

            temp_prices = np.zeros_like(S_valid[sigma_zero_mask])
            if np.any(call_mask_zero):
                temp_prices[call_mask_zero] = call_val_zero[call_mask_zero]
            if np.any(put_mask_zero):
                temp_prices[put_mask_zero] = put_val_zero[put_mask_zero]

            # Write back into prices
            idx_valid = np.flatnonzero(valid_mask)
            prices[idx_valid[sigma_zero_mask]] = temp_prices

        # Branch 2: Positive volatility -> standard BS with safe numerics
        if np.any(sigma_pos_mask):
            S_p = S_valid[sigma_pos_mask]
            K_p = K_valid[sigma_pos_mask]
            T_p = T_valid[sigma_pos_mask]
            r_p = r_valid[sigma_pos_mask]
            q_p = q_valid[sigma_pos_mask]
            sig_p = sigma_valid[sigma_pos_mask]

            # Safe log and denominator
            log_S_over_K = np.log(np.maximum(S_p / np.maximum(K_p, eps), eps))
            sqrt_T = np.sqrt(T_p)
            denom = sig_p * sqrt_T

            # Compute d1/d2 and clamp to avoid extreme tails
            d1 = (log_S_over_K + (r_p - q_p + 0.5 * sig_p**2) * T_p) / np.maximum(denom, eps)
            d2 = d1 - sig_p * sqrt_T
            d1 = np.clip(d1, -40.0, 40.0)
            d2 = np.clip(d2, -40.0, 40.0)

            exp_neg_qT = np.exp(-q_p * T_p)
            exp_neg_rT = np.exp(-r_p * T_p)

            call_mask_p = option_type_valid[sigma_pos_mask] == 'C'
            put_mask_p = option_type_valid[sigma_pos_mask] == 'P'

            temp_prices = np.zeros_like(S_p)
            if np.any(call_mask_p):
                temp_prices[call_mask_p] = (
                    S_p[call_mask_p] * exp_neg_qT[call_mask_p] * norm.cdf(d1[call_mask_p])
                    - K_p[call_mask_p] * exp_neg_rT[call_mask_p] * norm.cdf(d2[call_mask_p])
                )
            if np.any(put_mask_p):
                temp_prices[put_mask_p] = (
                    K_p[put_mask_p] * exp_neg_rT[put_mask_p] * norm.cdf(-d2[put_mask_p])
                    - S_p[put_mask_p] * exp_neg_qT[put_mask_p] * norm.cdf(-d1[put_mask_p])
                )

            # Write back
            idx_valid = np.flatnonzero(valid_mask)
            prices[idx_valid[sigma_pos_mask]] = temp_prices
    
    return prices


# DATA PROCESSING FUNCTIONS

def load_and_prepare_data(mode):
    """Load and prepare data based on the specified mode with comprehensive filtering"""
    print(f"Loading final options dataset for {mode} analysis...")
    
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Data file not found: {DATA_FILE}")
        return None
    
    df = pd.read_csv(DATA_FILE)
    print(f"Original dataset size: {len(df):,} rows")

    # Convert dates and calculate time to expiration
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    df['time_to_expiration'] = (df['exdate'] - df['date']).dt.days / 365.25
    df['days_to_maturity'] = (df['exdate'] - df['date']).dt.days
    
    config = MODES[mode]
    volatility_col = config['volatility_col']
    
    # Required columns based on mode
    required_cols = ['spx_close', 'strike_price', 'time_to_expiration', 
                    'risk_free_rate', volatility_col, 'dividend_rate', 
                    'mid_price', 'cp_flag', 'date', 'volume','best_bid', 'best_offer']
    
    # Remove columns that don't exist in the dataset
    existing_required_cols = [col for col in required_cols if col in df.columns]
    
    print(f"Initial dataset size: {len(df)}")
    
    # Apply percentage bid-ask spread filter FIRST
    if SPREAD_FILTER_ENABLED and {'best_bid','best_offer','mid_price'}.issubset(df.columns):
        initial_count = len(df)
        # Ensure positivity and valid mid to avoid divide-by-zero
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
    
    # Apply filtering options
    # Apply moneyness filtering (S/K ratio)
    if 'moneyness' in df.columns and (MONEYNESS_MIN is not None or MONEYNESS_MAX is not None):
        initial_count = len(df)
        if MONEYNESS_MIN is not None and MONEYNESS_MAX is not None:
            df = df[(df['moneyness'] >= MONEYNESS_MIN) & (df['moneyness'] <= MONEYNESS_MAX)]
            print(f"Applied moneyness filter ({MONEYNESS_MIN} ≤ S/K ≤ {MONEYNESS_MAX}): {len(df):,} rows (removed {initial_count - len(df):,})")
        
        elif MONEYNESS_MIN is not None:
            df = df[df['moneyness'] >= MONEYNESS_MIN]
            print(f"Applied moneyness filter (S/K ≥ {MONEYNESS_MIN}): {len(df):,} rows (removed {initial_count - len(df):,})")
        
        elif MONEYNESS_MAX is not None:
            df = df[df['moneyness'] <= MONEYNESS_MAX]
            print(f"Applied moneyness filter (S/K ≤ {MONEYNESS_MAX}): {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Apply time to maturity filtering
    if 'days_to_maturity' in df.columns and (DAYS_TO_MATURITY_MIN is not None or DAYS_TO_MATURITY_MAX is not None):
        initial_count = len(df)
        if DAYS_TO_MATURITY_MIN is not None and DAYS_TO_MATURITY_MAX is not None:
            df = df[(df['days_to_maturity'] >= DAYS_TO_MATURITY_MIN) & (df['days_to_maturity'] <= DAYS_TO_MATURITY_MAX)]
            print(f"Applied time to maturity filter ({DAYS_TO_MATURITY_MIN} ≤ days ≤ {DAYS_TO_MATURITY_MAX}): {len(df):,} rows (removed {initial_count - len(df):,})")
        elif DAYS_TO_MATURITY_MIN is not None:
            df = df[df['days_to_maturity'] >= DAYS_TO_MATURITY_MIN]
            print(f"Applied time to maturity filter (days ≥ {DAYS_TO_MATURITY_MIN}): {len(df):,} rows (removed {initial_count - len(df):,})")
        elif DAYS_TO_MATURITY_MAX is not None:
            df = df[df['days_to_maturity'] <= DAYS_TO_MATURITY_MAX]
            print(f"Applied time to maturity filter (days ≤ {DAYS_TO_MATURITY_MAX}): {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Apply quality filters for high-quality options data
    initial_count = len(df)
    
    # Filter 1: Ask > Bid (valid bid-ask spread)
    FILTER_VALID_SPREAD = False  # Match MLP2 default
    if FILTER_VALID_SPREAD:  
        valid_spread = df['best_offer'] > df['best_bid']
        df = df[valid_spread]
        print(f"Removed {initial_count - len(df)} options with invalid bid-ask spread (ask <= bid)")
    
    # Filter 2: Mid price >= 0.05 (minimum meaningful price)
    MIN_MEANINGFUL_PRICE = 0.05
    initial_count = len(df)
    df = df[df['mid_price'] >= MIN_MEANINGFUL_PRICE]
    print(f"Removed {initial_count - len(df)} options with mid price < ${MIN_MEANINGFUL_PRICE}")
    
    # Filter 3: Legacy spread ratio relative to mid (skip if percentage filter is enabled)
    if not SPREAD_FILTER_ENABLED and {'best_bid','best_offer','mid_price'}.issubset(df.columns):
        MAX_SPREAD_RATIO = 100  # Match MLP2 default
        initial_count = len(df)
        spread_ratio = (df['best_offer'] - df['best_bid']) / df['mid_price']
        df = df[spread_ratio <= MAX_SPREAD_RATIO]
        print(f"Removed {initial_count - len(df)} options with bid-ask spread ratio > {MAX_SPREAD_RATIO*100:.0f}% (legacy)")
    
    # Step 0: Remove rows with missing values in key columns
    initial_count = len(df)
    DEFAULT_FEATURES = ['spx_close', 'strike_price', 'days_to_maturity', 'risk_free_rate', 'dividend_rate', volatility_col]
    TARGET_COLUMN = 'mid_price'
    existing_required_cols = [col for col in DEFAULT_FEATURES + [TARGET_COLUMN] if col in df.columns]
    df = df.dropna(subset=existing_required_cols)
    print(f"After removing rows with missing values in key columns: {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Step 1: Filter out invalid and non-finite values (matching Black-Scholes)
    initial_count = len(df)
    
    is_finite_cols = (
        df['days_to_maturity'].replace([np.inf, -np.inf], np.nan).notna() &
        df['mid_price'].replace([np.inf, -np.inf], np.nan).notna() &
        df['strike_price'].replace([np.inf, -np.inf], np.nan).notna() &
        df['spx_close'].replace([np.inf, -np.inf], np.nan).notna() &
        df[volatility_col].replace([np.inf, -np.inf], np.nan).notna() &
        df['risk_free_rate'].replace([np.inf, -np.inf], np.nan).notna() &
        df['dividend_rate'].replace([np.inf, -np.inf], np.nan).notna()
    )

    # Basic validity filters
    valid_mask = (
        is_finite_cols &
        (df['days_to_maturity'] >= 0) &
        (df['mid_price'] > 0) &
        (df['strike_price'] > 0) &
        (df['spx_close'] > 0) &
        (df[volatility_col] >= 0)
    )
    
    # Additional filter for extreme volatilities; cap volatility to a reasonable maximum to avoid numerical issues
    MAX_REASONABLE_VOL = 5.0  # 500% annualized
    valid_mask = valid_mask & (df[volatility_col] <= MAX_REASONABLE_VOL)
    
    df = df[valid_mask]
    print(f"After filtering invalid values (including negative volatilities): {len(df):,} rows (removed {initial_count - len(df):,})")
    
    # Step 2: Zero volume filtering (moved here to match Black-Scholes order)
    if 'volume' in df.columns and ZERO_VOLUME_INCLUSION_RATE < 1.0:
        initial_count = len(df)
        zero_volume_mask = df['volume'] == 0
        non_zero_volume_df = df[~zero_volume_mask]
        zero_volume_df = df[zero_volume_mask]
        
        print(f"Zero volume options: {len(zero_volume_df):,} rows")
        print(f"Non-zero volume options: {len(non_zero_volume_df):,} rows")
        
        if ZERO_VOLUME_INCLUSION_RATE > 0.0 and len(zero_volume_df) > 0:
            # Sample the specified proportion of zero-volume options
            n_zero_volume_to_keep = int(len(zero_volume_df) * ZERO_VOLUME_INCLUSION_RATE)
            if n_zero_volume_to_keep > 0:
                zero_volume_sample = zero_volume_df.sample(n=n_zero_volume_to_keep, random_state=SEED)
                df = pd.concat([non_zero_volume_df, zero_volume_sample], ignore_index=True)
                print(f"Kept {n_zero_volume_to_keep:,} zero-volume options ({ZERO_VOLUME_INCLUSION_RATE:.1%})")
            else:
                df = non_zero_volume_df
                print("Excluded all zero-volume options")
        else:
            df = non_zero_volume_df
            print("Excluded all zero-volume options")
        
        print(f"Applied zero volume filter (inclusion rate: {ZERO_VOLUME_INCLUSION_RATE:.1%}): {len(df):,} rows (removed {initial_count - len(df):,})")
        # --- Reorder rows to avoid volume-block bias before splitting ---
        # If we did a time-based percent split, ensure chronological ordering.
        # Otherwise, shuffle to remove any concatenation-induced blocks.
        if USE_TIME_BASED_SPLIT and USE_TIME_PERCENT_SPLIT and 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            print("Re-sorted dataset by date to support time-based percent split.")
        else:
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            print("Shuffled dataset rows to avoid ordering bias during random split.")
    
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=SEED)
        print(f"Sampled {SAMPLE_SIZE} rows from {len(df)} available")
    
    print(f"Final dataset size: {len(df)}")
    
    # Remove unwanted columns (keep best_bid and best_offer for spread accuracy calculation)
    columns_to_remove = ['secid', 'exdate', 'spx_open', 'spx_high', 'spx_low']
    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if existing_columns_to_remove:
        df = df.drop(columns=existing_columns_to_remove)
        print(f"Removed columns: {', '.join(existing_columns_to_remove)}")
    
    print(f"\nFinal dataset: {len(df):,} options")
    
    if 'cp_flag' in df.columns:
        cp_counts = df['cp_flag'].value_counts()
        print(f"  - Calls: {cp_counts.get('C', 0):,}")
        print(f"  - Puts: {cp_counts.get('P', 0):,}")
    
    return df

def create_data_splits(X, y, df, indices):
    """
    EXACT COPY of MLP2's create_data_splits function.
    """
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

def create_random_splits(X, y, indices, df_original=None):
    
    df_view = df_original.iloc[indices].reset_index(drop=True)
    
    # Step 2: Decide stratify values (EXACT same logic as Black-Scholes)
    if USE_MULTI_DIM_STRATIFICATION:
        stratify_values = _build_multidim_strata(df_view)
    else:
        stratify_values = df_view['mid_price'].values
    
    # Step 3: Calculate ratios 
    test_val_total = 1.0 - TRAIN_RATIO
    val_fraction_of_temp = VALIDATION_RATIO / test_val_total if test_val_total > 0 else 0.5
    
    # Step 4: Create index array
    n_samples = len(df_view)
    all_idx = np.arange(n_samples)
    
    # Step 5: Numeric fallback 
    if np.issubdtype(np.array(stratify_values).dtype, np.number) and len(np.unique(stratify_values)) > 20:
        stratify_values = pd.qcut(stratify_values, q=5, labels=False, duplicates='drop')
    
    # Step 6: First split 
    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=test_val_total,
        random_state=SEED,
        shuffle=True,
        stratify=stratify_values,
    )
    
    # Step 7: Second split stratification   
    if USE_MULTI_DIM_STRATIFICATION:
        temp_view = df_view.iloc[temp_idx]
        temp_strat = _build_multidim_strata(temp_view)
    else:
        temp_strat = df_view.iloc[temp_idx]['mid_price'].values
    
    if np.issubdtype(np.array(temp_strat).dtype, np.number) and len(np.unique(temp_strat)) > 20:
        temp_strat = pd.qcut(temp_strat, q=3, labels=False, duplicates='drop')
    
    # Step 8: Second split 
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - val_fraction_of_temp,
        random_state=SEED,
        shuffle=True,
        stratify=temp_strat,
    )
    
    # Step 9: Map the local indices back to original indices
    # train_idx, val_idx, test_idx are currently indices into df_view
    # We need to map them back to the original indices
    idx_train = indices[train_idx]
    idx_val = indices[val_idx] 
    idx_test = indices[test_idx]
    
    # Step 10: Extract the actual data using original indices
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]  
    X_test, y_test = X[idx_test], y[idx_test]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test

def create_dataset_splits(df):
    """
    Wrapper function that adapts MLP2's create_data_splits to work with Black-Scholes DataFrame structure.
    """
    print("\nSplitting data...")
    
    # Reset index to ensure consistent splitting behavior
    df = df.reset_index(drop=True)
    
    # Create dummy X and y arrays and indices for compatibility with MLP2 function
    indices = np.arange(len(df))
    X = np.arange(len(df))  # Dummy X array
    y = df['mid_price'].values  # Use mid_price as dummy y
    
    # Call MLP2's exact splitting function
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = create_data_splits(X, y, df, indices)
    
    # Extract DataFrames using the indices
    train_df = df.iloc[idx_train].copy()
    val_df = df.iloc[idx_val].copy()
    test_df = df.iloc[idx_test].copy()
    
    print("Data split summary:")
    print(f"Train: {len(train_df):,} samples ({len(train_df)/len(df):.1%})")
    print(f"Val:   {len(val_df):,} samples ({len(val_df)/len(df):.1%})")
    print(f"Test:  {len(test_df):,} samples ({len(test_df)/len(df):.1%})")

    if 'date' in df.columns and len(train_df) > 0:
        print("\nDate ranges:")
        print(f"Train: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
        if len(val_df) > 0:
            print(f"Val:   {val_df['date'].min().date()} to {val_df['date'].max().date()}")
        if len(test_df) > 0:
            print(f"Test:  {test_df['date'].min().date()} to {test_df['date'].max().date()}")

    return train_df, val_df, test_df

def compute_black_scholes_prices(df, mode):
    """Compute Black-Scholes prices only (no Greeks)"""
    config = MODES[mode]
    volatility_col = config['volatility_col']
    
    print(f"Computing Black-Scholes prices using {volatility_col}...")
    
    # Extract parameters
    S = df['spx_close'].values
    K = df['strike_price'].values
    T = df['time_to_expiration'].values
    r = df['risk_free_rate'].values
    sigma = df[volatility_col].values
    option_type = df['cp_flag'].values
    q = df['dividend_rate'].values
    
    # Compute Black-Scholes prices only
    prices = black_scholes_price_only(S, K, T, r, sigma, option_type, q)
    
    # Add results to dataframe
    df['bs_price'] = prices
    
    # Calculate errors
    df['price_error'] = df['mid_price'] - df['bs_price']
    df['abs_price_error'] = np.abs(df['price_error'])
    df['pct_error'] = (df['price_error'] / df['mid_price']) * 100
    df['abs_pct_error'] = np.abs(df['pct_error'])
    
    print(f"   Black-Scholes calculation complete")
    return df

# ANALYSIS AND REPORTING

def create_analysis_plots(df, mode, output_dir):
    """Create comprehensive analysis plots with new layout"""
    config = MODES[mode]
    
    print("Creating analysis plots...")
    
    # Check if dataset is empty
    if len(df) == 0:
        print("   Warning: Empty dataset, skipping analysis plots")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Price comparison scatter plot (keep existing)
    plt.figure(figsize=(12, 8))
    plt.scatter(df['mid_price'], df['bs_price'], alpha=0.5, s=1)
    plt.plot([df['mid_price'].min(), df['mid_price'].max()], 
             [df['mid_price'].min(), df['mid_price'].max()], 'r--', lw=2)
    plt.xlabel('Market Price')
    plt.ylabel('Black-Scholes Price')
    plt.title(f'Market vs Black-Scholes Prices\n{config["description"]}')
    plt.grid(True, alpha=0.3)
    
    # Add R² to plot
    correlation = np.corrcoef(df['mid_price'], df['bs_price'])[0, 1]
    r_squared = correlation ** 2
    plt.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'price_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Error Analysis (2x2 layout) - Error vs Moneyness with price and percentage errors
    plt.figure(figsize=(16, 12))
    
    # TOP LEFT: Absolute Price Error vs Moneyness (color coded by days to maturity)
    plt.subplot(2, 2, 1)
    if 'moneyness' in df.columns:
        scatter1 = plt.scatter(df['moneyness'], df['abs_price_error'], 
                            c=df['days_to_maturity'], alpha=0.6, s=1, cmap='viridis')
        plt.xlabel('Moneyness (S/K)', fontsize=12)
        plt.ylabel('Absolute Price Error ($)', fontsize=12)
        plt.title('Absolute Price Error vs Moneyness', fontsize=14, fontweight='bold')
        # Expanded range - show more datapoints
        plt.ylim(0, df['abs_price_error'].max() * 1.1)
        plt.xlim(df['moneyness'].min() * 0.95, df['moneyness'].max() * 1.05)
        plt.colorbar(scatter1, label='Days to Maturity')
        plt.grid(True, alpha=0.3)
    
    # TOP RIGHT: Non-absolute Price Error vs Moneyness (color coded by days to maturity)
    plt.subplot(2, 2, 2)
    if 'moneyness' in df.columns:
        scatter2 = plt.scatter(df['moneyness'], df['price_error'], 
                            c=df['days_to_maturity'], alpha=0.6, s=1, cmap='plasma')
        plt.xlabel('Moneyness (S/K)', fontsize=12)
        plt.ylabel('Price Error ($)', fontsize=12)
        plt.title('Price Error vs Moneyness', fontsize=14, fontweight='bold')
        # Expanded range - show more datapoints
        plt.ylim(df['price_error'].min() * 1.1, df['price_error'].max() * 1.1)
        plt.xlim(df['moneyness'].min() * 0.95, df['moneyness'].max() * 1.05)
        plt.colorbar(scatter2, label='Days to Maturity')
        plt.grid(True, alpha=0.3)
    
    # BOTTOM LEFT: Absolute Percentage Error vs Moneyness (color coded by days to maturity)
    plt.subplot(2, 2, 3)
    if 'moneyness' in df.columns:
        scatter3 = plt.scatter(df['moneyness'], df['abs_pct_error'], 
                            c=df['days_to_maturity'], alpha=0.6, s=1, cmap='coolwarm')
        plt.xlabel('Moneyness (S/K)', fontsize=12)
        plt.ylabel('Absolute Percentage Error (%)', fontsize=12)
        plt.title('Absolute Percentage Error vs Moneyness', fontsize=14, fontweight='bold')
        # Expanded range - show more datapoints
        plt.ylim(0, df['abs_pct_error'].max() * 1.1)
        plt.xlim(df['moneyness'].min() * 0.95, df['moneyness'].max() * 1.05)
        plt.colorbar(scatter3, label='Days to Maturity')
        plt.grid(True, alpha=0.3)
    
    # BOTTOM RIGHT: Percentage Error vs Moneyness (color coded by days to maturity)
    plt.subplot(2, 2, 4)
    if 'moneyness' in df.columns:
        scatter4 = plt.scatter(df['moneyness'], df['pct_error'], 
                            c=df['days_to_maturity'], alpha=0.6, s=1, cmap='RdYlBu')
        plt.xlabel('Moneyness (S/K)', fontsize=12)
        plt.ylabel('Percentage Error (%)', fontsize=12)
        plt.title('Percentage Error vs Moneyness', fontsize=14, fontweight='bold')
        # Expanded range - show more datapoints
        plt.ylim(df['pct_error'].min() * 1.1, df['pct_error'].max() * 1.1)
        plt.xlim(df['moneyness'].min() * 0.95, df['moneyness'].max() * 1.05)
        plt.colorbar(scatter4, label='Days to Maturity')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create new MLP-style plots
    create_residuals_histograms(df, output_dir)
    create_residuals_histograms_frequency(df, output_dir)
    create_residuals_vs_features(df, output_dir)
    create_mae_analysis(df, output_dir)
    create_mape_analysis(df, output_dir)
    
    # 5. Create extra plot (Q-Q and Delta distribution)
    create_extra_plot(df, output_dir)
    
    print(f"Analysis plots saved to {output_dir}")

def create_residuals_histograms(df, output_dir, use_density=True):
    # Create histograms for residuals analysis with configurable y-axis
    hist_type = "density" if use_density else "frequency"
    print(f"Creating residuals histograms ({hist_type})...")
    
    if len(df) == 0:
        print(f"Warning: Empty dataset, skipping residuals histograms ({hist_type})")
        return
    
    residuals = df['price_error']
    abs_residuals = np.abs(residuals)
    pct_error = df['pct_error']
    abs_pct_error = df['abs_pct_error']
    
    plt.figure(figsize=(15, 10))
    
    # 1. Absolute Residuals Distribution
    plt.subplot(2, 2, 1)
    abs_res_95th = np.percentile(abs_residuals, 95)
    abs_res_filtered = abs_residuals[abs_residuals <= abs_res_95th]
    plt.hist(abs_res_filtered, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=use_density)
    plt.xlabel('Absolute Residuals ($)', fontsize=12)
    plt.ylabel('Density' if use_density else 'Frequency', fontsize=12)
    plt.title('Absolute Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    plt.subplot(2, 2, 2)
    res_5th, res_95th = np.percentile(residuals, [5, 95])
    res_filtered = residuals[(residuals >= res_5th) & (residuals <= res_95th)]
    plt.hist(res_filtered, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=use_density)
    plt.xlabel('Residuals ($)', fontsize=12)
    plt.ylabel('Density' if use_density else 'Frequency', fontsize=12)
    plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Percentage Residuals Distribution
    plt.subplot(2, 2, 3)
    pct_error_filtered = pct_error[(pct_error >= -100) & (pct_error <= 100)]
    plt.hist(pct_error_filtered, bins=50, alpha=0.7, color='orange', edgecolor='black', density=use_density)
    plt.xlabel('Percentage Error (%)', fontsize=12)
    plt.ylabel('Density' if use_density else 'Frequency', fontsize=12)
    plt.title('Percentage Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Absolute Percentage Error Distribution
    plt.subplot(2, 2, 4)
    abs_pct_error_filtered = abs_pct_error[abs_pct_error <= 100]
    plt.hist(abs_pct_error_filtered, bins=50, alpha=0.7, color='purple', edgecolor='black', density=use_density)
    plt.xlabel('Absolute Percentage Error (%)', fontsize=12)
    plt.ylabel('Density' if use_density else 'Frequency', fontsize=12)
    plt.title('Absolute Percentage Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    filename = 'residuals_histograms.png' if use_density else 'residuals_histograms_frequency.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Residuals histograms ({hist_type}) saved to {save_path}")

def create_residuals_histograms_frequency(df, output_dir):
    create_residuals_histograms(df, output_dir, use_density=False)

def create_residuals_vs_features(df, output_dir):
    """Create residuals vs features scatter plots - Second PNG (matching MLP style)"""
    print("Creating residuals vs features plots...")
    
    if len(df) == 0:
        print("   Warning: Empty dataset, skipping residuals vs features plots")
        return
    
    residuals = df['price_error']
    fitted_values = df['bs_price']
    days_to_maturity = df['days_to_maturity']
    
    plt.figure(figsize=(16, 12))
    
    # 1. Residuals vs Fitted Values (color coded by days to maturity)
    plt.subplot(2, 2, 1)
    scatter1 = plt.scatter(fitted_values, residuals, c=days_to_maturity, alpha=0.3, s=1, cmap='viridis')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values (BS Price $)', fontsize=12)
    plt.ylabel('Residuals ($)', fontsize=12)
    plt.title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter1, label='Days to Maturity')
    
    # 2. Average Absolute Residuals vs Days to Maturity (binned bar chart)
    plt.subplot(2, 2, 2)
    # Create small bins for days to maturity
    days_bins = np.linspace(days_to_maturity.min(), days_to_maturity.max(), 50)
    days_binned = pd.cut(days_to_maturity, bins=days_bins)
    
    # Calculate average absolute residuals and count for each bin
    abs_residuals = np.abs(residuals)
    binned_avg_residuals = []
    bin_centers = []
    bin_counts = []
    
    for i in range(len(days_bins)-1):
        bin_mask = (days_to_maturity >= days_bins[i]) & (days_to_maturity < days_bins[i+1])
        if bin_mask.sum() > 0:  # Only include bins with data
            binned_avg_residuals.append(abs_residuals[bin_mask].mean())
            bin_centers.append((days_bins[i] + days_bins[i+1]) / 2)
            bin_counts.append(bin_mask.sum())
    
    # Calculate bin width for bar chart
    bin_width = (days_bins[1] - days_bins[0]) * 0.8
    # Color-code bars based on number of data points
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    norm = colors.Normalize(vmin=min(bin_counts), vmax=max(bin_counts))
    cmap = cm.get_cmap('viridis')
    bar_colors = [cmap(norm(count)) for count in bin_counts]
    bars = plt.bar(bin_centers, binned_avg_residuals, width=bin_width, 
                   color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    plt.xlabel('Days to Maturity', fontsize=12)
    plt.ylabel('Average Absolute Residuals ($)', fontsize=12)
    plt.title('Avg Abs Residuals vs Days to Maturity', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Create colorbar manually
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Data Points Count')
    
    # 3. Percentage Error vs Fitted Values (color coded by days to maturity)
    plt.subplot(2, 2, 3)
    pct_error = df['pct_error']
    scatter3 = plt.scatter(fitted_values, pct_error, c=days_to_maturity, alpha=0.3, s=1, cmap='plasma')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values (BS Price $)', fontsize=12)
    plt.ylabel('Percentage Error (%)', fontsize=12)
    plt.title('Percentage Error vs Fitted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter3, label='Days to Maturity')
    
    # 4. Average Absolute Residuals vs Historical Volatility (binned bar chart)
    plt.subplot(2, 2, 4)
    if 'historical_volatility' in df.columns:
        # Create small bins for historical volatility
        vol_bins = np.linspace(df['historical_volatility'].min(), df['historical_volatility'].max(), 50)
        
        # Calculate average absolute residuals and count for each bin
        vol_binned_avg_residuals = []
        vol_bin_centers = []
        vol_bin_counts = []
        
        for i in range(len(vol_bins)-1):
            bin_mask = (df['historical_volatility'] >= vol_bins[i]) & (df['historical_volatility'] < vol_bins[i+1])
            if bin_mask.sum() > 0:  # Only include bins with data
                vol_binned_avg_residuals.append(abs_residuals[bin_mask].mean())
                vol_bin_centers.append((vol_bins[i] + vol_bins[i+1]) / 2)
                vol_bin_counts.append(bin_mask.sum())
        
        # Calculate bin width for bar chart
        vol_bin_width = (vol_bins[1] - vol_bins[0]) * 0.8
        # Color-code bars based on number of data points
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        vol_norm = colors.Normalize(vmin=min(vol_bin_counts), vmax=max(vol_bin_counts))
        vol_cmap = cm.get_cmap('plasma')
        vol_bar_colors = [vol_cmap(vol_norm(count)) for count in vol_bin_counts]
        vol_bars = plt.bar(vol_bin_centers, vol_binned_avg_residuals, width=vol_bin_width, 
                          color=vol_bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.xlabel('Historical Volatility', fontsize=12)
        plt.ylabel('Average Absolute Residuals ($)', fontsize=12)
        plt.title('Avg Abs Residuals vs Historical Volatility', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        # Create colorbar manually
        vol_sm = cm.ScalarMappable(cmap=vol_cmap, norm=vol_norm)
        vol_sm.set_array([])
        plt.colorbar(vol_sm, ax=plt.gca(), label='Data Points Count')
    else:
        plt.text(0.5, 0.5, 'Historical volatility data not available', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'residuals_vs_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Residuals vs features plots saved")

def create_mae_analysis(df, output_dir):
    """Create Mean Absolute Error analysis by features"""
    print("Creating MAE analysis by features...")
    
    if len(df) == 0:
        print("   Warning: Empty dataset, skipping MAE analysis")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. MAE by Price Quantiles
    ax1 = axes[0, 0]
    price_quantiles = pd.qcut(df['mid_price'], q=5, labels=False, duplicates='drop')
    mae_by_quantile = []
    quantile_labels = []
    quantile_counts = []
    
    # Generate labels with actual price bounds
    unique_quantiles = sorted(price_quantiles[~pd.isna(price_quantiles)].unique())
    for q in unique_quantiles:
        mask = price_quantiles == q
        if mask.sum() > 0:
            subset_prices = df[mask]['mid_price']
            min_price = subset_prices.min()
            max_price = subset_prices.max()
            mae_by_quantile.append(np.mean(np.abs(df[mask]['price_error'])))
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
    if 'moneyness' in df.columns:
        df['moneyness_bucket'] = get_moneyness_buckets(df['moneyness'])
        moneyness_analysis = df.groupby('moneyness_bucket').agg({
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
    df['tte_bucket'] = get_time_buckets(df['time_to_expiration'])
    tte_analysis = df.groupby('tte_bucket').agg({
        'price_error': lambda x: np.mean(np.abs(x)),
        'time_to_expiration': 'count'
    }).reset_index()
    tte_analysis.columns = ['tte_bucket', 'mae', 'count']
    
    bars3 = ax3.bar(range(len(tte_analysis)), tte_analysis['mae'], 
                   color='lightgreen', alpha=0.8, edgecolor='black')
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
    if 'historical_volatility' in df.columns:
        # Create volatility quintiles
        vol_quantiles = pd.qcut(df['historical_volatility'], q=5, labels=False, duplicates='drop')
        mae_by_vol = []
        vol_labels = []
        vol_counts = []
        
        # Generate labels with actual volatility bounds
        unique_quantiles = sorted(vol_quantiles[~pd.isna(vol_quantiles)].unique())
        for q in unique_quantiles:
            mask = vol_quantiles == q
            if mask.sum() > 0:
                subset_vol = df[mask]['historical_volatility']
                min_vol = subset_vol.min()
                max_vol = subset_vol.max()
                mae_by_vol.append(np.mean(np.abs(df[mask]['price_error'])))
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
    
    # 5. MAE by Volume
    ax5 = axes[2, 0]
    if 'volume' in df.columns:
        df['volume_bucket'] = get_volume_buckets(df['volume'])
        volume_analysis = df.groupby('volume_bucket').agg({
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
    if 'equity_uncertainty' in df.columns:
        df['uncertainty_bucket'] = get_uncertainty_buckets(df['equity_uncertainty'])
        uncertainty_analysis = df.groupby('uncertainty_bucket').agg({
            'price_error': lambda x: np.mean(np.abs(x)),
            'equity_uncertainty': 'count'
        }).reset_index()
        uncertainty_analysis.columns = ['uncertainty_bucket', 'mae', 'count']
        
        bars6 = ax6.bar(range(len(uncertainty_analysis)), uncertainty_analysis['mae'], 
                       color='gold', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars6, uncertainty_analysis['count'])):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
        ax6.set_xticks(range(len(uncertainty_analysis)))
        ax6.set_xticklabels(uncertainty_analysis['uncertainty_bucket'], rotation=45, ha='right')
    else:
        ax6.text(0.5, 0.5, 'Equity uncertainty data not available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
    ax6.set_title('Mean Absolute Error by Equity Uncertainty', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Mean Absolute Error ($)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'mae_by_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("MAE by features analysis saved")

def create_mape_analysis(df, output_dir):
    """Create MAPE analysis by features"""
    print("Creating MAPE analysis by features...")
    
    if len(df) == 0:
        print("   Warning: Empty dataset, skipping MAPE analysis")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    
    # 1. MAPE by Price Quantiles
    ax1 = axes[0, 0]
    price_quantiles = pd.qcut(df['mid_price'], q=5, labels=False, duplicates='drop')
    mape_by_quantile = []
    quantile_labels = []
    quantile_counts = []
    
    # Generate labels with actual price bounds
    unique_quantiles = sorted(price_quantiles[~pd.isna(price_quantiles)].unique())
    for q in unique_quantiles:
        mask = price_quantiles == q
        if mask.sum() > 0:
            subset_prices = df[mask]['mid_price']
            min_price = subset_prices.min()
            max_price = subset_prices.max()
            mape_by_quantile.append(np.mean(df[mask]['abs_pct_error']))
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
    if 'moneyness' in df.columns:
        df['moneyness_bucket'] = get_moneyness_buckets(df['moneyness'])
        moneyness_analysis = df.groupby('moneyness_bucket').agg({
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
    df['tte_bucket'] = get_time_buckets(df['time_to_expiration'])
    tte_analysis = df.groupby('tte_bucket').agg({
        'abs_pct_error': 'mean',
        'time_to_expiration': 'count'
    }).reset_index()
    tte_analysis.columns = ['tte_bucket', 'mape', 'count']
    
    bars3 = ax3.bar(range(len(tte_analysis)), tte_analysis['mape'], 
                   color='lightgreen', alpha=0.8, edgecolor='black')
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
    if 'historical_volatility' in df.columns:
        vol_quantiles = pd.qcut(df['historical_volatility'], q=5, labels=False, duplicates='drop')
        mape_by_vol = []
        vol_labels = []
        vol_counts = []
        
        # Generate labels with actual volatility bounds
        unique_quantiles = sorted(vol_quantiles[~pd.isna(vol_quantiles)].unique())
        for q in unique_quantiles:
            mask = vol_quantiles == q
            if mask.sum() > 0:
                subset_vol = df[mask]['historical_volatility']
                min_vol = subset_vol.min()
                max_vol = subset_vol.max()
                mape_by_vol.append(np.mean(df[mask]['abs_pct_error']))
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
    
    # 5. MAPE by Volume
    ax5 = axes[2, 0]
    if 'volume' in df.columns:
        df['volume_bucket'] = get_volume_buckets(df['volume'])
        volume_analysis = df.groupby('volume_bucket').agg({
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
    if 'equity_uncertainty' in df.columns:
        df['uncertainty_bucket'] = get_uncertainty_buckets(df['equity_uncertainty'])
        uncertainty_analysis = df.groupby('uncertainty_bucket').agg({
            'abs_pct_error': 'mean',
            'equity_uncertainty': 'count'
        }).reset_index()
        uncertainty_analysis.columns = ['uncertainty_bucket', 'mape', 'count']
        
        bars6 = ax6.bar(range(len(uncertainty_analysis)), uncertainty_analysis['mape'], 
                       color='gold', alpha=0.8, edgecolor='black')
        for i, (bar, count) in enumerate(zip(bars6, uncertainty_analysis['count'])):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'n={count:,}', ha='center', va='bottom', fontsize=8)
        ax6.set_xticks(range(len(uncertainty_analysis)))
        ax6.set_xticklabels(uncertainty_analysis['uncertainty_bucket'], rotation=45, ha='right')
    else:
        ax6.text(0.5, 0.5, 'Equity uncertainty data not available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
    ax6.set_title('MAPE by Equity Uncertainty', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'mape_by_features.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("MAPE by features analysis saved")

def create_extra_plot(df, output_dir):
    """Create extra plots: Q-Q plot and Delta distribution - Extra PNG"""
    print("Creating extra plots (Q-Q and Delta distribution)...")
    
    # Check if dataset is empty
    if len(df) == 0:
        print("   Warning: Empty dataset, skipping extra plots")
        return
    
    residuals = df['price_error']
    
    plt.figure(figsize=(15, 6))
    
    # 1. Q-Q Plot for Normality of Residuals
    plt.subplot(1, 2, 1)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Residuals Normality)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'extra_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Extra plots saved")

def create_diagnostic_plots(df, output_dir):
    # Create diagnostic plots with updated layout (matching MLP style)
    print("Creating diagnostic plots...")
    
    # Check if dataset is empty
    if len(df) == 0:
        print("Warning: Empty dataset, skipping diagnostic plots")
        return
    
    residuals = df['price_error']
    fitted_values = df['bs_price']
    
    plt.figure(figsize=(15, 10))
    
    # 1. Price Comparison (1st plot) - True vs Predicted
    plt.subplot(2, 2, 1)
    plt.scatter(df['mid_price'], fitted_values, alpha=0.3, s=1, color='blue')
    min_val, max_val = 0, max(df['mid_price'].max(), fitted_values.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('True Price ($)', fontsize=12)
    plt.ylabel('Black-Scholes Price ($)', fontsize=12)
    plt.title('True vs Black-Scholes Prices', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Error Distribution (3rd plot) - next to 1st
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Q-Q Plot (third position)
    plt.subplot(2, 2, 3)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Residuals Normality)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Residuals vs Historical Volatility (instead of residuals distribution)
    plt.subplot(2, 2, 4)
    if 'historical_volatility' in df.columns:
        plt.scatter(df['historical_volatility'], residuals, alpha=0.3, s=1, color='orange')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Historical Volatility', fontsize=12)
        plt.ylabel('Residuals ($)', fontsize=12)
        plt.title('Residuals vs Historical Volatility', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'diagnostic_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Diagnostic plots saved")

def generate_analysis_report(df, mode, output_dir):
    """Generate comprehensive analysis report similar to OLS regression"""
    config = MODES[mode]
    
    print("Generating comprehensive analysis report...")
    
    # Check if dataset is empty
    if len(df) == 0:
        print("   Warning: Empty dataset, generating minimal report")
        report_content = f"""

BLACK-SCHOLES ANALYSIS - COMPREHENSIVE REPORT
{'=' * 60}

Configuration:
- Analysis Type: {config['description']}
- Volatility Source: {config['volatility_col']}
- Sample Size Limit: {SAMPLE_SIZE if SAMPLE_SIZE else 'None (all data)'}
- Zero Volume Inclusion Rate: {ZERO_VOLUME_INCLUSION_RATE:.1%}

Data Filtering Applied:
- Moneyness Filter: {MONEYNESS_MIN} ≤ S/K ≤ {MONEYNESS_MAX}
- Time to Maturity Filter: {DAYS_TO_MATURITY_MIN} ≤ days ≤ {DAYS_TO_MATURITY_MAX}

ERROR: No data remaining after filtering!
Please adjust the filtering parameters to include more data.
"""

        report_path = os.path.join(output_dir, f'{mode}_analysis_summary.txt')
        with open(report_path, 'w') as f:
            f.write(report_content)
        print(f"   Error report saved to {report_path}")
        return report_content
    
    # Calculate summary statistics
    price_error_stats = df['price_error'].describe()
    pct_error_stats = df['pct_error'].describe()
    abs_pct_error_stats = df['abs_pct_error'].describe()
    
    # Calculate additional metrics
    correlation = np.corrcoef(df['mid_price'], df['bs_price'])[0, 1]
    r_squared = correlation ** 2
    
    # Mean absolute error and RMSE
    mae = np.mean(np.abs(df['price_error']))
    rmse = np.sqrt(np.mean(df['price_error'] ** 2))
    mse = np.mean(df['price_error'] ** 2)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs(df['pct_error']))
    
    # Calculate bid-ask spread accuracy (% of predictions within bid-ask spread)
    bid_prices = df['best_bid'].values
    ask_prices = df['best_offer'].values
    bs_prices = df['bs_price'].values
    
    # Check if Black-Scholes predictions are within bid-ask spread
    in_spread = (bs_prices >= bid_prices) & (bs_prices <= ask_prices)
    spread_accuracy = np.mean(in_spread) * 100
    
    # Option type breakdown
    call_data = df[df['cp_flag'] == 'C']
    put_data = df[df['cp_flag'] == 'P']
    
    # Statistical tests on residuals
    from scipy import stats
    residuals = df['price_error']
    
    # Normality test (Shapiro-Wilk on sample if too large)
    if len(residuals) > 5000:
        sample_residuals = np.random.choice(residuals, 5000, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(sample_residuals)
        normality_note = " (tested on 5000 random sample)"
    else:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        normality_note = ""
    
    # Jarque-Bera test for normality
    jb_stat, jb_p = stats.jarque_bera(residuals)
    
    report_content = f"""
BLACK-SCHOLES ANALYSIS - COMPREHENSIVE REPORT
{'=' * 60}

Configuration:
- Analysis Type: {config['description']}
- Volatility Source: {config['volatility_col']}
- Sample Size Limit: {SAMPLE_SIZE if SAMPLE_SIZE else 'None (all data)'}
- Zero Volume Inclusion Rate: {ZERO_VOLUME_INCLUSION_RATE:.1%}

Data Filtering Applied:
- Moneyness Filter: {MONEYNESS_MIN} ≤ S/K ≤ {MONEYNESS_MAX}
- Time to Maturity Filter: {DAYS_TO_MATURITY_MIN} ≤ days ≤ {DAYS_TO_MATURITY_MAX}

Final Dataset (Test Subset):
- Total Options: {len(df):,}
- Calls: {len(call_data):,} ({len(call_data)/len(df)*100:.1f}%)
- Puts: {len(put_data):,} ({len(put_data)/len(df)*100:.1f}%)

MODEL PERFORMANCE METRICS (Test Subset):
- Mean Squared Error (MSE): {mse:.4f}
- Mean Absolute Error (MAE): ${mae:.4f}
- Root Mean Square Error (RMSE): ${rmse:.4f}
- Mean Absolute Percentage Error (MAPE): {mape:.2f}%
- Correlation (Market vs BS): {correlation:.4f}
- R-squared: {r_squared:.4f}
- Bid-Ask Spread Accuracy (Test): {spread_accuracy:.2f}%

RESIDUAL ANALYSIS:
- Mean Residual: ${np.mean(residuals):.4f}
- Std Residual: ${np.std(residuals):.4f}
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

PRICE ERROR STATISTICS (Market - Black-Scholes):
{price_error_stats.to_string()}

PERCENTAGE ERROR STATISTICS:
{pct_error_stats.to_string()}

ABSOLUTE PERCENTAGE ERROR STATISTICS:
{abs_pct_error_stats.to_string()}

OPTION TYPE COMPARISON:
Calls ({len(call_data):,} options):
- Mean Abs Error: ${np.mean(np.abs(call_data['price_error'])):.4f}
- RMSE: ${np.sqrt(np.mean(call_data['price_error'] ** 2)):.4f}
- Mean Abs % Error: {np.mean(call_data['abs_pct_error']):.2f}%
- R-squared: {np.corrcoef(call_data['mid_price'], call_data['bs_price'])[0, 1]**2:.4f}

Puts ({len(put_data):,} options):
- Mean Abs Error: ${np.mean(np.abs(put_data['price_error'])):.4f}
- RMSE: ${np.sqrt(np.mean(put_data['price_error'] ** 2)):.4f}
- Mean Abs % Error: {np.mean(put_data['abs_pct_error']):.2f}%
- R-squared: {np.corrcoef(put_data['mid_price'], put_data['bs_price'])[0, 1]**2:.4f}

MONEYNESS ANALYSIS:
"""    
    # Add moneyness breakdown if available
    if 'moneyness' in df.columns:
        # Define moneyness buckets (centralized)
        df['moneyness_bucket'] = pd.cut(df['moneyness'], bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)
        
        moneyness_analysis = df.groupby('moneyness_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        moneyness_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        moneyness_analysis = moneyness_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
        
        report_content += f"\nMoneyness Bucket Analysis:\n{moneyness_analysis.to_string()}\n"
    
    # Add time to expiration analysis (centralized)
    df['tte_bucket'] = pd.cut(df['time_to_expiration'], bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)
    
    tte_analysis = df.groupby('tte_bucket').agg({
        'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
        'abs_pct_error': 'mean'
    }).round(4)
    tte_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
    tte_analysis = tte_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
    
    report_content += f"\nTime to Expiration Analysis:\n{tte_analysis.to_string()}\n"
    
    # Add volume-based analysis
    if 'volume' in df.columns:
        df['volume_bucket'] = get_volume_buckets(df['volume'])
        volume_analysis = df.groupby('volume_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        volume_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        volume_analysis = volume_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
        report_content += f"\nVolume Bucket Analysis:\n{volume_analysis.to_string()}\n"
    
    # Add historical volatility analysis
    if 'historical_volatility' in df.columns:
        df['hv_bucket'] = get_hv_buckets(df['historical_volatility'])
        
        # Get quantile ranges for display
        hv_quantiles = df['historical_volatility'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        hv_ranges = {}
        for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
            min_val = hv_quantiles.iloc[i]
            max_val = hv_quantiles.iloc[i+1]
            hv_ranges[q] = f"Q{i+1} ({min_val:.3f} - {max_val:.3f})"
        
        hv_analysis = df.groupby('hv_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        hv_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        hv_analysis = hv_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
        
        # Rename index with ranges
        hv_analysis.index = [hv_ranges.get(str(idx), str(idx)) for idx in hv_analysis.index]
        report_content += f"\nHistorical Volatility Bucket Analysis:\n{hv_analysis.to_string()}\n"
    
    # Add price range analysis
    df['price_bucket'] = get_price_buckets(df['mid_price'])
    
    # Get quantile ranges for display
    price_quantiles = df['mid_price'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    price_ranges = {}
    for i, q in enumerate(['Q1', 'Q2', 'Q3', 'Q4', 'Q5']):
        min_val = price_quantiles.iloc[i]
        max_val = price_quantiles.iloc[i+1]
        price_ranges[q] = f"Q{i+1} (${min_val:.2f} - ${max_val:.2f})"
    
    price_analysis = df.groupby('price_bucket').agg({
        'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
        'abs_pct_error': 'mean'
    }).round(4)
    price_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
    price_analysis = price_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
    
    # Rename index with ranges
    price_analysis.index = [price_ranges.get(str(idx), str(idx)) for idx in price_analysis.index]
    report_content += f"\nPrice Range Analysis:\n{price_analysis.to_string()}\n"
    
    # Add equity uncertainty analysis
    if 'equity_uncertainty' in df.columns:
        df['uncertainty_bucket'] = get_uncertainty_buckets(df['equity_uncertainty'])
        uncertainty_analysis = df.groupby('uncertainty_bucket').agg({
            'price_error': ['count', lambda x: np.mean(np.abs(x)), 'median', 'std'],
            'abs_pct_error': 'mean'
        }).round(4)
        uncertainty_analysis.columns = ['Count', 'MAE', 'Median', 'SD', 'MAPE']
        uncertainty_analysis = uncertainty_analysis[['Count', 'MAE', 'MAPE', 'Median', 'SD']]
        report_content += f"\nEquity Uncertainty Analysis:\n{uncertainty_analysis.to_string()}\n"
    
    report_content += f"""

INTERPRETATION:
{'=' * 20}
{config.get('interpretation', 'Standard Black-Scholes theoretical pricing analysis.')}

Note: This analysis uses {config['volatility_col']} for volatility estimation.
{'WARNING: This is a circular validation check using implied volatility derived from market prices.' if mode == 'implied' else 'This represents theoretical model performance using historical volatility.'}
"""
    
    # Save report
    report_path = os.path.join(output_dir, f'{mode}_analysis_summary.txt')
    with open(report_path, 'w') as f:
        f.write(report_content)
        
        f.write("FILES GENERATED:\n")
        f.write(f"  - {config['output_file']} (enhanced dataset)\n")
        f.write(f"  - price_comparison.png (market vs BS prices)\n")
        f.write(f"  - error_analysis.png (comprehensive error analysis)\n")
        f.write(f"  - analysis_report.txt (this report)\n")
    
    print(f"   Analysis report saved to {report_path}")

def save_enhanced_dataset(df, mode, output_dir):
    """Save the enhanced dataset"""
    config = MODES[mode]
    output_file = os.path.join(output_dir, config['output_file'])
    
    df.to_csv(output_file, index=False)
    print(f"Enhanced dataset saved to: {output_file}")
    
    print(f"   Columns added: bs_price, price_error, abs_price_error, pct_error, abs_pct_error")

# MAIN EXECUTION

def main():
    """Main function"""
    # Use the ANALYSIS_MODE variable from configuration
    mode = ANALYSIS_MODE
    
    # Ensure consistent plotting style
    setup_plot_style()

    # Validate mode
    if mode not in MODES:
        print(f"Error: Invalid mode '{mode}'. Must be 'historical' or 'implied'.")
        print(f"Please change the ANALYSIS_MODE variable at the top of the script.")
        return
    
    config = MODES[mode]
    
    print("="*60)
    print(f"BLACK-SCHOLES ANALYSIS - {mode.upper()} MODE")
    print(f"Description: {config['description']}")
    print("="*60)
    
    # Create output directory (optionally timestamped)
    base_output_dir = os.path.join(BLACK_SCHOLES_DIR, config['output_dir'])
    if USE_TIMESTAMPED_OUTPUT:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_output_dir, f"run_{run_id}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Results will be saved to: {output_dir}")
    else:
        output_dir = base_output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")
    
    # Load and prepare data - EXACT MLP2 approach
    df_original = load_and_prepare_data(mode)
    if df_original is None:
        return
    
    # Create X and y arrays - EXACT MLP2 approach
    X = df_original[['strike_price', 'spx_close', 'days_to_maturity']].values
    y = df_original['mid_price'].values
    
    # Split data using improved method (time-based or stratified random) - EXACT MLP2 approach
    indices = np.arange(len(X))
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = create_data_splits(
        X, y, df_original, indices)
    
    # Add verification prints to confirm matching splits with MLP2 - EXACT MLP2 approach
    temp_idx = np.concatenate([idx_val, idx_test])
    print("First/last train date:", df_original.iloc[idx_train]['date'].min(),
          df_original.iloc[idx_train]['date'].max())
    print("First/last temp date :", df_original.iloc[temp_idx]['date'].min(),
          df_original.iloc[temp_idx]['date'].max())
    print("Counts -> train/temp/val/test:",
          len(idx_train), len(temp_idx), len(idx_val), len(idx_test))
    
    # Save test identifiers for cross-script verification - EXACT MLP2 approach
    test_identifiers = df_original.iloc[idx_test][['date', 'cp_flag', 'strike_price', 'spx_close', 'days_to_maturity', 'mid_price']]
    test_identifiers_path = os.path.join(output_dir, 'test_identifiers.csv')
    test_identifiers.to_csv(test_identifiers_path, index=False)
    
    # Save indices for cross-script verification
    np.save(os.path.join(output_dir, 'test_indices.npy'), idx_test)
    
    temp_identifiers = df_original.iloc[temp_idx][['date', 'cp_flag', 'strike_price', 'spx_close', 'days_to_maturity', 'mid_price']]
    temp_identifiers_path = os.path.join(output_dir, 'temp_identifiers.csv')
    temp_identifiers.to_csv(temp_identifiers_path, index=False)
    
    print("Saved test_indices.npy, temp_identifiers.csv and test_identifiers.csv for verification")
    
    # Use test subset for Black-Scholes analysis
    df_test = df_original.iloc[idx_test].copy()
    print(f"\nUsing test subset for Black-Scholes analysis: {len(df_test):,} rows")
    
    # Compute Black-Scholes prices & create analysis plots on test subset only
    df_test = compute_black_scholes_prices(df_test, mode)
    create_analysis_plots(df_test, mode, output_dir)
    
    # Generate report & save enhanced dataset for test subset only
    generate_analysis_report(df_test, mode, output_dir)
    save_enhanced_dataset(df_test, mode, output_dir)
    
    # If using historical mode, also update the main final_options_dataset.csv with new columns
    if mode == 'historical':
        # Do not overwrite the full main dataset from a subset; skip bulk update to avoid inconsistencies
        print(f"\n Note: Skipping main dataset overwrite since analysis is restricted to test subset.")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    if USE_TIMESTAMPED_OUTPUT:
        print(f"Run ID: {run_id}")
    if mode == 'historical':
        print(f"Main dataset enhanced with Black-Scholes calculations")

if __name__ == "__main__":
    main()
