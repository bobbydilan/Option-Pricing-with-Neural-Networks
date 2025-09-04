import os
import glob
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Numerical stability constant
EPS = 1e-8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

RESULT_DIRS = {
    'Regression': os.path.join(PROJECT_ROOT, '02Regression', 'results'),
    'BlackScholes': os.path.join(PROJECT_ROOT, '03Black_Scholes', 'results'),
    'MLP1': os.path.join(PROJECT_ROOT, '04MLP1', 'results'),
    'MLP2': os.path.join(PROJECT_ROOT, '05MLP2', 'results'),
}

OUTPUT_DIR = os.path.join(BASE_DIR, 'results', f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def latest_subdir(path: str) -> Optional[str]:
    if not os.path.isdir(path):
        return None
    subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def latest_subdir_with_file(path: str, filename: str) -> Optional[str]:
    """Return the newest subdirectory under path that contains filename.

    Falls back to newest subdir even if file missing, but prioritizes those containing the file.
    """
    if not os.path.isdir(path):
        return None
    subdirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if not subdirs:
        return None
    # Sort newest first
    subdirs.sort(key=os.path.getmtime, reverse=True)
    for d in subdirs:
        if os.path.exists(os.path.join(d, filename)):
            return d
    # If none contain the file, return newest anyway
    return subdirs[0]

def load_regression(latest_dir: str) -> Optional[pd.DataFrame]:
    """Load OLS predictions from latest regression results folder.
    Expects file: options_with_ols_predictions.csv
    Returns normalized DataFrame with columns: ['y_true','y_pred','moneyness','hist_volatility','days_to_maturity','model']
    """
    
    csv_path = os.path.join(latest_dir, 'options_with_ols_predictions.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    cols = {}
    # Map columns
    cols['y_true'] = df.get('mid_price')
    cols['y_pred'] = df.get('ols_pred')
    # Fallbacks (if alternative naming)
    if cols['y_true'] is None and 'actual_mid_price' in df.columns:
        cols['y_true'] = df['actual_mid_price']
    if cols['y_pred'] is None and 'predicted_mid_price' in df.columns:
        cols['y_pred'] = df['predicted_mid_price']

    # Features
    if 'moneyness' in df.columns:
        cols['moneyness'] = df['moneyness']
    else:
        # Try compute if possible
        if 'spx_close' in df.columns and 'strike_price' in df.columns:
            cols['moneyness'] = df['spx_close'] / df['strike_price'].replace(0, np.nan)
        else:
            cols['moneyness'] = np.nan

    if 'historical_volatility' in df.columns:
        cols['hist_volatility'] = df['historical_volatility']
    else:
        cols['hist_volatility'] = df.get('hist_vol_30d', np.nan)

    cols['days_to_maturity'] = df.get('days_to_maturity', np.nan)

    out = pd.DataFrame(cols)
    out['model'] = 'Regression'
    return out.dropna(subset=['y_true', 'y_pred'])


def load_black_scholes(latest_dir: str) -> Optional[pd.DataFrame]:
    """Load Black-Scholes predictions from latest results folder.
    Expects file: options_with_black_scholes_historical.csv (column bs_price)
    """
    csv_path = os.path.join(latest_dir, 'options_with_black_scholes_historical.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    cols = {
        'y_true': df.get('mid_price'),
        'y_pred': df.get('bs_price'),
        'moneyness': df.get('moneyness', np.nan),
        'hist_volatility': df.get('historical_volatility', np.nan),
        'days_to_maturity': df.get('days_to_maturity', np.nan),
    }
    out = pd.DataFrame(cols)
    out['model'] = 'Black-Scholes'
    return out.dropna(subset=['y_true', 'y_pred'])


def load_mlp_predictions(csv_path: str, model_name: str) -> Optional[pd.DataFrame]:
    """Generic loader for MLP predictions CSV created by our training scripts.
    Handles different column variants between MLP1 and MLP2.
    """
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)

    # Identify true/pred columns
    y_true = None
    y_pred = None
    for cand in ['actual_mid_price', 'mid_price', 'actual_price']:
        if cand in df.columns:
            y_true = df[cand]
            break
    for cand in ['predicted_mid_price', 'y_pred', 'prediction', 'mlp_pred']:
        if cand in df.columns:
            y_pred = df[cand]
            break

    # Moneyness
    if 'moneyness' in df.columns:
        moneyness = df['moneyness']
    else:
        spx_cols = [c for c in df.columns if 'spx_close' in c]
        k_cols = [c for c in df.columns if 'strike_price' in c]
        if spx_cols and k_cols:
            spx = df[spx_cols[0]].astype(float)
            k = df[k_cols[0]].astype(float).replace(0, np.nan)
            moneyness = spx / k
        else:
            moneyness = np.nan

    # Historical vol
    hv_col = None
    for cand in ['historical_volatility_input', 'historical_volatility', 'hist_vol_30d']:
        if cand in df.columns:
            hv_col = cand
            break
    hist_volatility = df[hv_col] if hv_col else np.nan

    # Days to maturity
    dtm_col = None
    for cand in ['days_to_maturity_input', 'days_to_maturity']:
        if cand in df.columns:
            dtm_col = cand
            break
    days_to_maturity = df[dtm_col] if dtm_col else np.nan

    out = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'moneyness': moneyness,
        'hist_volatility': hist_volatility,
        'days_to_maturity': days_to_maturity,
    })
    out['model'] = model_name
    return out.dropna(subset=['y_true', 'y_pred'])


def load_mlp1(latest_dir: str) -> Optional[pd.DataFrame]:
    return load_mlp_predictions(os.path.join(latest_dir, 'mlp_predictions.csv'), 'MLP1')


def load_mlp2(latest_dir: str) -> Optional[pd.DataFrame]:
    return load_mlp_predictions(os.path.join(latest_dir, 'mlp_predictions.csv'), 'MLP2')


def get_all_model_frames() -> Dict[str, pd.DataFrame]:
    frames = {}
    # Regression
    reg_dir = latest_subdir_with_file(RESULT_DIRS['Regression'], 'options_with_ols_predictions.csv')
    if reg_dir:
        df = load_regression(reg_dir)
        if df is not None and not df.empty:
            frames['Regression'] = df
    # Black-Scholes
    bs_dir = latest_subdir_with_file(RESULT_DIRS['BlackScholes'], 'options_with_black_scholes_historical.csv')
    if bs_dir:
        df = load_black_scholes(bs_dir)
        if df is not None and not df.empty:
            frames['Black-Scholes'] = df
    # MLP1
    mlp1_dir = latest_subdir_with_file(RESULT_DIRS['MLP1'], 'mlp_predictions.csv')
    if mlp1_dir:
        df = load_mlp1(mlp1_dir)
        if df is not None and not df.empty:
            frames['MLP1'] = df
    # MLP2
    mlp2_dir = latest_subdir_with_file(RESULT_DIRS['MLP2'], 'mlp_predictions.csv')
    if mlp2_dir:
        df = load_mlp2(mlp2_dir)
        if df is not None and not df.empty:
            frames['MLP2'] = df
    return frames


def compute_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['residual'] = df['y_pred'] - df['y_true']
    df['abs_error'] = np.abs(df['residual'])
    den = np.maximum(np.abs(df['y_true'].values), EPS)
    df['pct_error'] = 100.0 * df['residual'].values / den
    df['abs_pct_error'] = np.abs(df['pct_error'])
    return df


COLORS = {
    'Regression': '#1f77b4',
    'Black-Scholes': '#ff7f0e',
    'MLP1': '#2ca02c',
    'MLP2': '#d62728',
}


def plot_pred_vs_true(frames: Dict[str, pd.DataFrame], out_dir: str):
    plt.figure(figsize=(10, 8))
    max_val = 0
    for name, df in frames.items():
        s = df.sample(min(len(df), 50000), random_state=42) if len(df) > 50000 else df
        plt.scatter(s['y_true'], s['y_pred'], s=6, alpha=0.35, c=COLORS.get(name, None), label=name)
        max_val = max(max_val, s[['y_true', 'y_pred']].max().max())
    lim = (0, max_val * 1.05)
    plt.plot(lim, lim, 'k--', linewidth=1, label='Ideal')
    plt.xlim(lim)
    plt.ylim(lim)
    plt.xlabel('True Price')
    plt.ylabel('Predicted Price')
    plt.title('Predicted vs True Price (All Models)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, 'pred_vs_true_all_models.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_residuals_vs_true(frames: Dict[str, pd.DataFrame], out_dir: str):
    plt.figure(figsize=(10, 8))
    for name, df in frames.items():
        s = df.sample(min(len(df), 50000), random_state=42) if len(df) > 50000 else df
        plt.scatter(s['y_true'], s['residual'], s=6, alpha=0.35, c=COLORS.get(name, None), label=name)
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('True Price')
    plt.ylabel('Residual (Pred - True)')
    plt.title('Residuals vs True Price (All Models)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(out_dir, 'residuals_vs_true_all_models.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def grouped_bar(values: Dict[str, np.ndarray], labels: np.ndarray, title: str, ylabel: str, filename: str, out_dir: str):
    models = list(values.keys())
    n_bins = len(labels)
    x = np.arange(n_bins)
    width = 0.8 / max(len(models), 1)

    plt.figure(figsize=(14, 8))
    for i, m in enumerate(models):
        plt.bar(x + i * width, values[m], width=width, color=COLORS.get(m, None), alpha=0.8, label=m)
    plt.xticks(x + width * (len(models) - 1) / 2, labels, rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()


def compute_bin_stats(frames: Dict[str, pd.DataFrame], by: str, bins: Optional[np.ndarray] = None, q: Optional[int] = None,
                      label_fmt: str = '{:.2f}-{:.2f}', use_quantiles: bool = False) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute mean absolute error per bin for each model.
    - by: column name to bin on (y_true, moneyness, hist_volatility, time_years)
    - bins: explicit bin edges (if provided)
    - q: number of quantiles (if use_quantiles=True)
    Returns: (model_to_values, labels)
    """
    out = {}
    labels_arr = None
    for name, df in frames.items():
        s = df.copy()
        if by == 'time_years' and 'time_years' not in s.columns:
            s['time_years'] = s['days_to_maturity'] / 365.25
        target = s[by].values
        metric = s['abs_error'].values

        # Build bins
        if use_quantiles and q is not None:
            try:
                cats = pd.qcut(target, q=q, labels=False, duplicates='drop')
                # labels from quantile edges
                q_edges = pd.qcut(target, q=q, retbins=True, duplicates='drop')[1]
                labels = np.array([label_fmt.format(q_edges[i], q_edges[i+1]) for i in range(len(q_edges)-1)])
            except Exception:
                # Fallback to fixed bins if quantiles fail
                if bins is None:
                    raise
                cats = pd.cut(target, bins=bins, labels=False, include_lowest=True)
                labels = np.array([label_fmt.format(bins[i], bins[i+1]) for i in range(len(bins)-1)])
        else:
            if bins is None:
                raise ValueError('bins must be provided when not using quantiles')
            cats = pd.cut(target, bins=bins, labels=False, include_lowest=True)
            labels = np.array([label_fmt.format(bins[i], bins[i+1]) for i in range(len(bins)-1)])

        # Aggregate mean absolute error by bin
        vals = []
        for b in range(len(labels)):
            mask = cats == b
            if np.any(mask):
                vals.append(np.nanmean(metric[mask]))
            else:
                vals.append(np.nan)
        out[name] = np.array(vals)
        labels_arr = labels
    return out, labels_arr


def main():
    frames = get_all_model_frames()
    if not frames:
        print('No model results found. Make sure you have run models and produced CSV outputs.')
        return

    # Compute errors
    frames = {name: compute_errors(df) for name, df in frames.items()}

    # Save metadata of which folders were used (prefer those containing the expected file)
    used = {
        'Regression': latest_subdir_with_file(RESULT_DIRS['Regression'], 'options_with_ols_predictions.csv'),
        'Black-Scholes': latest_subdir_with_file(RESULT_DIRS['BlackScholes'], 'options_with_black_scholes_historical.csv'),
        'MLP1': latest_subdir_with_file(RESULT_DIRS['MLP1'], 'mlp_predictions.csv'),
        'MLP2': latest_subdir_with_file(RESULT_DIRS['MLP2'], 'mlp_predictions.csv'),
    }
    with open(os.path.join(OUTPUT_DIR, 'sources.json'), 'w') as f:
        json.dump(used, f, indent=2)

    # Core comparison plots
    plot_pred_vs_true(frames, OUTPUT_DIR)
    plot_residuals_vs_true(frames, OUTPUT_DIR)

    # Error by Price Quantiles (use quantiles of y_true)
    price_values, price_labels = compute_bin_stats(frames, by='y_true', q=5, use_quantiles=True,
                                                   label_fmt='{:.2f}-{:.2f}')
    grouped_bar(price_values, price_labels, 'Mean Absolute Error by Price Quantile', 'MAE ($)', 'mae_by_price_quantile.png', OUTPUT_DIR)

    # Error by Moneyness
    mny_bins = np.array([0.0, 0.8, 0.95, 1.05, 1.2, np.inf])
    mny_values, mny_labels = compute_bin_stats(frames, by='moneyness', bins=mny_bins, use_quantiles=False,
                                               label_fmt='{:.2f}-{:.2f}')
    grouped_bar(mny_values, mny_labels, 'Mean Absolute Error by Moneyness Bin', 'MAE ($)', 'mae_by_moneyness.png', OUTPUT_DIR)

    # Error by Historical Volatility (quantiles)
    hv_values, hv_labels = compute_bin_stats(frames, by='hist_volatility', q=4, use_quantiles=True,
                                             label_fmt='{:.3f}-{:.3f}')
    grouped_bar(hv_values, hv_labels, 'Mean Absolute Error by Historical Volatility Quantile', 'MAE ($)', 'mae_by_hist_vol.png', OUTPUT_DIR)

    # Error by Time to Maturity in years (fixed bins)
    t_bins = np.array([0.0, 0.08, 0.25, 0.5, 1.0, np.inf])
    t_values, t_labels = compute_bin_stats(frames, by='time_years', bins=t_bins, use_quantiles=False,
                                           label_fmt='{:.2f}-{:.2f}')
    grouped_bar(t_values, t_labels, 'Mean Absolute Error by Time to Maturity (Years)', 'MAE ($)', 'mae_by_ttm.png', OUTPUT_DIR)

    print(f'Comparison plots written to: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
