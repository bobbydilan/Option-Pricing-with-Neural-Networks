import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
import json
from typing import Dict, Optional
from datetime import datetime

# Set Times New Roman as default font
plt.rcParams['font.family'] = 'Times New Roman'

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

OUTPUT_DIR = os.path.join(BASE_DIR, 'results', f'diagnostic_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    'Regression': '#1f77b4',
    'Black-Scholes': '#ff7f0e',
    'MLP1-1': '#2ca02c',
    'MLP1-2': '#32b332',
    'MLP1-3': '#3cc63c',
    'MLP2-1': '#d62728',
    'MLP2-2': '#e03838',
    'MLP2-3': '#ea4848',
}

def latest_subdir_with_file(path: str, filename: str) -> Optional[str]:
    """Return the newest subdirectory under path that contains filename."""
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

def parse_diagnostic_file(file_path: str) -> Dict[str, Dict[str, float]]:
    """Parse comprehensive_diagnostics.txt file and extract MAE by category."""
    if not os.path.exists(file_path):
        return {}
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = {}
    
    # Parse Price Quantiles - get ALL quantiles
    price_section = re.search(r'--- STATISTICS BY PRICE QUANTILES ---\n(.*?)(?=\n--- |\Z)', content, re.DOTALL)
    if price_section:
        price_data = {}
        matches = re.findall(r'Q(\d+) \(\$([^)]+)\):\n  Count: [^\n]+\n  MAE: \$([0-9.]+)', price_section.group(1))
        for q_num, price_range, mae in matches:
            price_data[f'Q{q_num}'] = float(mae)
        results['price_quantiles'] = price_data
        print(f"  Price quantiles: {len(price_data)} categories")
    
    # Parse Moneyness - get ALL categories
    moneyness_section = re.search(r'--- STATISTICS BY MONEYNESS ---\n(.*?)(?=\n--- |\Z)', content, re.DOTALL)
    if moneyness_section:
        moneyness_data = {}
        matches = re.findall(r'(OTM|ATM|ITM) \([^)]+\):\n  Count: [^\n]+\n  MAE: \$([0-9.]+)', moneyness_section.group(1))
        for category, mae in matches:
            moneyness_data[category] = float(mae)
        results['moneyness'] = moneyness_data
        print(f"  Moneyness: {len(moneyness_data)} categories")
    
    # Parse Time to Maturity - get only the main TTM buckets (not sub-categories)
    ttm_section = re.search(r'--- STATISTICS BY TIME TO MATURITY ---\n(.*?)(?=\n--- |\Z)', content, re.DOTALL)
    if ttm_section:
        ttm_data = {}
        # Match the TTM patterns: ≤1M, 1-3M, 3-6M, 6-9M, 9-12M, >12M
        matches = re.findall(r'^([≤>]?\d*-?\d*M[^(]*) \([^)]+\):\n  Count: [^\n]+\n  MAE: \$([0-9.]+)', ttm_section.group(1), re.MULTILINE)
        for category, mae in matches:
            category_clean = category.strip()
            ttm_data[category_clean] = float(mae)
        results['time_to_maturity'] = ttm_data
        print(f"  Time to maturity: {len(ttm_data)} categories")
    
    # Parse Historical Volatility - get ALL quantiles
    hv_section = re.search(r'--- STATISTICS BY HISTORICAL VOLATILITY ---\n(.*?)(?=\n--- |\Z)', content, re.DOTALL)
    if hv_section:
        hv_data = {}
        matches = re.findall(r'Q(\d+) \(([^)]+)\):\n  Count: [^\n]+\n  MAE: \$([0-9.]+)', hv_section.group(1))
        for q_num, range_val, mae in matches:
            hv_data[f'Q{q_num}'] = float(mae)
        results['historical_volatility'] = hv_data
        print(f"  Historical volatility: {len(hv_data)} categories")
    
    # Parse Volume - get ALL volume categories
    volume_section = re.search(r'--- STATISTICS BY VOLUME ---\n(.*?)(?=\n--- |\Z)', content, re.DOTALL)
    if volume_section:
        volume_data = {}
        matches = re.findall(r'(Zero|Low|Medium|High)(?:\s+\([^)]*\))?:\n  Count: [^\n]+\n  MAE: \$([0-9.]+)', volume_section.group(1))
        for category, mae in matches:
            volume_data[category] = float(mae)
        results['volume'] = volume_data
        print(f"  Volume: {len(volume_data)} categories")
    
    # Parse Equity Uncertainty (EPU) - get ALL categories
    epu_section = re.search(r'--- STATISTICS BY EQUITY UNCERTAINTY ---\n(.*?)(?=\n--- |\Z)', content, re.DOTALL)
    if epu_section:
        epu_data = {}
        matches = re.findall(r'(Low|Medium|High|Very High) \([^)]+\):\n  Count: [^\n]+\n  MAE: \$([0-9.]+)', epu_section.group(1))
        for category, mae in matches:
            epu_data[category] = float(mae)
        results['equity_uncertainty'] = epu_data
        print(f"  Equity uncertainty: {len(epu_data)} categories")
    
    return results

def compute_bucket_stats_from_csv(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute bucket statistics from a DataFrame with predictions."""
    results = {}
    
    # Compute errors
    df = df.copy()
    df['abs_error'] = np.abs(df['y_pred'] - df['y_true'])
    
    # Price Quantiles
    try:
        df['price_quantile'] = pd.qcut(df['y_true'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        price_stats = df.groupby('price_quantile')['abs_error'].mean()
        results['price_quantiles'] = price_stats.to_dict()
    except:
        results['price_quantiles'] = {}
    
    # Moneyness
    if 'moneyness' in df.columns:
        df['moneyness_bucket'] = pd.cut(df['moneyness'], 
                                       bins=[0, 0.9, 1.1, np.inf], 
                                       labels=['OTM', 'ATM', 'ITM'])
        moneyness_stats = df.groupby('moneyness_bucket')['abs_error'].mean()
        results['moneyness'] = moneyness_stats.to_dict()
    else:
        results['moneyness'] = {}
    
    # Time to Maturity
    if 'days_to_maturity' in df.columns:
        df['ttm_bucket'] = pd.cut(df['days_to_maturity'], 
                                 bins=[0, 30, 91, 182, 274, 365, np.inf],
                                 labels=['≤1M', '1-3M', '3-6M', '6-9M', '9-12M', '>12M'])
        ttm_stats = df.groupby('ttm_bucket')['abs_error'].mean()
        results['time_to_maturity'] = ttm_stats.to_dict()
    else:
        results['time_to_maturity'] = {}
    
    # Historical Volatility
    if 'hist_volatility' in df.columns:
        try:
            df['hv_quantile'] = pd.qcut(df['hist_volatility'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            hv_stats = df.groupby('hv_quantile')['abs_error'].mean()
            results['historical_volatility'] = hv_stats.to_dict()
        except:
            results['historical_volatility'] = {}
    else:
        results['historical_volatility'] = {}
    
    # Volume
    if 'volume' in df.columns:
        df['volume_bucket'] = pd.cut(df['volume'], 
                                    bins=[-1, 0, 100, 1000, np.inf],
                                    labels=['Zero', 'Low', 'Medium', 'High'])
        volume_stats = df.groupby('volume_bucket')['abs_error'].mean()
        results['volume'] = volume_stats.to_dict()
    else:
        results['volume'] = {}
    
    # EPU Index
    if 'epu_index' in df.columns:
        df['epu_bucket'] = pd.cut(df['epu_index'], 
                                 bins=[0, 50, 100, 150, np.inf],
                                 labels=['Low', 'Medium', 'High', 'Very High'])
        epu_stats = df.groupby('epu_bucket')['abs_error'].mean()
        results['equity_uncertainty'] = epu_stats.to_dict()
    else:
        results['equity_uncertainty'] = {}
    
    return results

def load_baseline_models() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load baseline models (Regression, Black-Scholes) using CSV method."""
    baseline_data = {}
    
    # Load Regression
    reg_dir = latest_subdir_with_file(RESULT_DIRS['Regression'], 'options_with_ols_predictions.csv')
    if reg_dir:
        csv_path = os.path.join(reg_dir, 'options_with_ols_predictions.csv')
        if os.path.exists(csv_path):
            print("Loading Regression model...")
            df = pd.read_csv(csv_path)
            # Map columns
            df_clean = pd.DataFrame({
                'y_true': df.get('mid_price', df.get('actual_mid_price')),
                'y_pred': df.get('ols_pred', df.get('predicted_mid_price')),
                'moneyness': df.get('moneyness'),
                'days_to_maturity': df.get('days_to_maturity'),
                'hist_volatility': df.get('historical_volatility', df.get('hist_vol_30d')),
                'volume': df.get('volume'),
                'epu_index': df.get('epu_index')
            })
            df_clean = df_clean.dropna(subset=['y_true', 'y_pred'])
            baseline_data['Regression'] = compute_bucket_stats_from_csv(df_clean)
            print(f"  Regression: {len(baseline_data['Regression'])} analysis types")
    
    # Load Black-Scholes
    bs_dir = latest_subdir_with_file(RESULT_DIRS['BlackScholes'], 'options_with_black_scholes_historical.csv')
    if bs_dir:
        csv_path = os.path.join(bs_dir, 'options_with_black_scholes_historical.csv')
        if os.path.exists(csv_path):
            print("Loading Black-Scholes model...")
            df = pd.read_csv(csv_path)
            # Map columns
            df_clean = pd.DataFrame({
                'y_true': df.get('mid_price'),
                'y_pred': df.get('bs_price'),
                'moneyness': df.get('moneyness'),
                'days_to_maturity': df.get('days_to_maturity'),
                'hist_volatility': df.get('historical_volatility'),
                'volume': df.get('volume'),
                'epu_index': df.get('epu_index')
            })
            df_clean = df_clean.dropna(subset=['y_true', 'y_pred'])
            baseline_data['Black-Scholes'] = compute_bucket_stats_from_csv(df_clean)
            print(f"  Black-Scholes: {len(baseline_data['Black-Scholes'])} analysis types")
    
    return baseline_data

def load_individual_predictions() -> Dict[str, pd.DataFrame]:
    """Load individual prediction data for scatter plots."""
    frames = {}
    
    # Load Regression
    reg_dir = latest_subdir_with_file(RESULT_DIRS['Regression'], 'options_with_ols_predictions.csv')
    if reg_dir:
        csv_path = os.path.join(reg_dir, 'options_with_ols_predictions.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_clean = pd.DataFrame({
                'y_true': df.get('mid_price', df.get('actual_mid_price')),
                'y_pred': df.get('ols_pred', df.get('predicted_mid_price')),
            })
            df_clean = df_clean.dropna(subset=['y_true', 'y_pred'])
            df_clean['abs_error'] = np.abs(df_clean['y_pred'] - df_clean['y_true'])
            frames['Regression'] = df_clean
            print(f"  Regression predictions: {len(df_clean)} samples")
    
    # Load Black-Scholes
    bs_dir = latest_subdir_with_file(RESULT_DIRS['BlackScholes'], 'options_with_black_scholes_historical.csv')
    if bs_dir:
        csv_path = os.path.join(bs_dir, 'options_with_black_scholes_historical.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_clean = pd.DataFrame({
                'y_true': df.get('mid_price'),
                'y_pred': df.get('bs_price'),
            })
            df_clean = df_clean.dropna(subset=['y_true', 'y_pred'])
            df_clean['abs_error'] = np.abs(df_clean['y_pred'] - df_clean['y_true'])
            frames['Black-Scholes'] = df_clean
            print(f"  Black-Scholes predictions: {len(df_clean)} samples")
    
    # Load MLP1 runs
    mlp1_runs = ['run_MLP1.1', 'run_MLP1.2', 'run_MLP1.3']
    for run_name in mlp1_runs:
        run_dir = os.path.join(RESULT_DIRS['MLP1'], run_name)
        csv_path = os.path.join(run_dir, 'mlp_predictions.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_clean = pd.DataFrame({
                'y_true': df.get('actual_mid_price'),
                'y_pred': df.get('predicted_mid_price'),
            })
            df_clean = df_clean.dropna(subset=['y_true', 'y_pred'])
            df_clean['abs_error'] = np.abs(df_clean['y_pred'] - df_clean['y_true'])
            model_name = f'MLP1-{run_name.split(".")[1]}'
            frames[model_name] = df_clean
            print(f"  {model_name} predictions: {len(df_clean)} samples")
    
    # Load MLP2 runs
    mlp2_runs = ['run_MLP2.1', 'run_MLP2.2', 'run_MLP2.3']
    for run_name in mlp2_runs:
        run_dir = os.path.join(RESULT_DIRS['MLP2'], run_name)
        csv_path = os.path.join(run_dir, 'mlp_predictions.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df_clean = pd.DataFrame({
                'y_true': df.get('actual_mid_price'),
                'y_pred': df.get('predicted_mid_price'),
            })
            df_clean = df_clean.dropna(subset=['y_true', 'y_pred'])
            df_clean['abs_error'] = np.abs(df_clean['y_pred'] - df_clean['y_true'])
            model_name = f'MLP2-{run_name.split(".")[1]}'
            frames[model_name] = df_clean
            print(f"  {model_name} predictions: {len(df_clean)} samples")
    
    return frames

def plot_predictions_vs_true(frames: Dict[str, pd.DataFrame]):
    """Create predicted vs true price scatter plot for all models."""
    plt.figure(figsize=(14, 10))
    
    for model, df in frames.items():
        # Sample data for plotting (to avoid overcrowding)
        sample_size = min(1000000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        plt.scatter(df_sample['y_true'], df_sample['y_pred'], 
                   alpha=0.6, s=1, color=COLORS.get(model, 'gray'), label=model)
    
    # Add perfect prediction line
    max_val = max([df['y_true'].max() for df in frames.values()])
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.8, label='Ideal')
    
    plt.xlabel('True Price', fontsize=34, fontweight='bold')
    plt.ylabel('Predicted Price', fontsize=34, fontweight='bold')
    plt.title('Predicted vs True Price (All Models)', fontsize=30, fontweight='bold', pad=50)
    plt.tick_params(axis='both', which='major', labelsize=30)  # Make tick labels bigger
    plt.legend(fontsize=18, markerscale=10)  # Make legend markers 10x bigger
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'pred_vs_true_all_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created plot: pred_vs_true_all_models.png")

def plot_residuals_vs_true(frames: Dict[str, pd.DataFrame]):
    """Create residuals vs true price scatter plot for all models."""
    plt.figure(figsize=(14, 10))
    
    for model, df in frames.items():
        # Sample data for plotting (to avoid overcrowding)
        sample_size = min(1000000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)
        
        residuals = df_sample['y_pred'] - df_sample['y_true']
        plt.scatter(df_sample['y_true'], residuals, 
                   alpha=0.6, s=1, color=COLORS.get(model, 'gray'), label=model)
    
    # Add zero line
    max_val = max([df['y_true'].max() for df in frames.values()])
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    
    plt.xlabel('True Price', fontsize=30, fontweight='bold')
    plt.ylabel('Residual (Pred - True)', fontsize=30, fontweight='bold')
    plt.title('Residuals vs True Price (All Models)', fontsize=30, fontweight='bold', pad=50)
    plt.tick_params(axis='both', which='major', labelsize=24)  # Make tick labels bigger
    plt.legend(fontsize=18, markerscale=10)  # Make legend markers 10x bigger
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'residuals_vs_true_all_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Created plot: residuals_vs_true_all_models.png")

def load_all_diagnostic_data() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load diagnostic data from all models."""
    all_data = {}
    
    # Load baseline models first
    baseline_data = load_baseline_models()
    all_data.update(baseline_data)
    
    # Load MLP1 runs
    mlp1_runs = ['run_MLP1.1', 'run_MLP1.2', 'run_MLP1.3']
    for run_name in mlp1_runs:
        run_dir = os.path.join(RESULT_DIRS['MLP1'], run_name)
        diag_file = os.path.join(run_dir, 'comprehensive_diagnostics.txt')
        if os.path.exists(diag_file):
            model_name = f'MLP1-{run_name.split(".")[1]}'
            all_data[model_name] = parse_diagnostic_file(diag_file)
            print(f"Loaded {model_name}: {len(all_data[model_name])} analysis types")
    
    # Load MLP2 runs
    mlp2_runs = ['run_MLP2.1', 'run_MLP2.2', 'run_MLP2.3']
    for run_name in mlp2_runs:
        run_dir = os.path.join(RESULT_DIRS['MLP2'], run_name)
        diag_file = os.path.join(run_dir, 'comprehensive_diagnostics.txt')
        if os.path.exists(diag_file):
            model_name = f'MLP2-{run_name.split(".")[1]}'
            all_data[model_name] = parse_diagnostic_file(diag_file)
            print(f"Loaded {model_name}: {len(all_data[model_name])} analysis types")
    
    return all_data

def create_comparison_plot(data: Dict[str, Dict[str, Dict[str, float]]], 
                          analysis_type: str, 
                          title: str, 
                          filename: str):
    """Create a grouped bar chart for a specific analysis type."""
    if not data or analysis_type not in next(iter(data.values())):
        print(f"Warning: No data available for {analysis_type}. Skipping plot.")
        return
    
    # Extract data for this analysis type - get all unique categories from all models
    models = list(data.keys())
    all_categories = set()
    for model in models:
        if analysis_type in data[model]:
            all_categories.update(data[model][analysis_type].keys())
    
    # Sort categories appropriately
    if analysis_type == 'time_to_maturity':
        # Custom order for TTM categories
        ttm_order = ['≤1M', '1-3M', '3-6M', '6-9M', '9-12M', '>12M']
        categories = [cat for cat in ttm_order if cat in all_categories]
        # Add any unexpected categories at the end
        categories.extend([cat for cat in sorted(all_categories) if cat not in ttm_order])
    elif analysis_type == 'volume':
        # Custom order for Volume categories
        volume_order = ['Zero', 'Low', 'Medium', 'High']
        categories = [cat for cat in volume_order if cat in all_categories]
        # Add any unexpected categories at the end
        categories.extend([cat for cat in sorted(all_categories) if cat not in volume_order])
    elif analysis_type == 'moneyness':
        # Custom order for Moneyness categories
        moneyness_order = ['OTM', 'ATM', 'ITM']
        categories = [cat for cat in moneyness_order if cat in all_categories]
        # Add any unexpected categories at the end
        categories.extend([cat for cat in sorted(all_categories) if cat not in moneyness_order])
    elif analysis_type == 'equity_uncertainty':
        # Custom order for Equity Uncertainty categories
        uncertainty_order = ['Low', 'Medium', 'High', 'Very High']
        categories = [cat for cat in uncertainty_order if cat in all_categories]
        # Add any unexpected categories at the end
        categories.extend([cat for cat in sorted(all_categories) if cat not in uncertainty_order])
    else:
        categories = sorted(list(all_categories))
    
    if not categories:
        print(f"Warning: No categories found for {analysis_type}. Skipping plot.")
        return
    
    # Create data matrix
    mae_values = {}
    for model in models:
        if analysis_type in data[model]:
            mae_values[model] = [data[model][analysis_type].get(cat, np.nan) for cat in categories]
        else:
            mae_values[model] = [np.nan] * len(categories)
    
    # Create plot
    n_cats = len(categories)
    x = np.arange(n_cats)
    width = 0.8 / len(models)
    
    plt.figure(figsize=(24, 12))
    
    # Set log scale BEFORE plotting
    plt.yscale('log')
    
    bars_list = []
    
    for i, model in enumerate(models):
        values = mae_values[model]
        # Filter out NaN values for plotting
        plot_values = [v if not np.isnan(v) and v > 0 else 0.01 for v in values]  # Use small positive value for log scale
        
        bars = plt.bar(x + i * width, plot_values, width=width, 
                      color=COLORS.get(model, f'C{i}'), alpha=0.8, label=model)
        bars_list.append(bars)
        
        # Add value labels on bars - show actual MAE values
        for j, (bar, original_value) in enumerate(zip(bars, values)):
            if not np.isnan(original_value) and original_value > 0:
                # Get the actual bar height (which is the log-scaled value)
                bar_top = bar.get_height()
                bar_bottom = 0.01  # Minimum value for log scale
                
                # Calculate proportional font size based on bar height (log scale)
                # Map bar height to font size range (12-24)
                log_height = np.log10(bar_top) - np.log10(bar_bottom)
                max_log_height = 3  # Assume max is around 1000 (log10(1000) = 3)
                font_size = 12 + (log_height / max_log_height) * 12  # Range 12-24
                font_size = max(12, min(24, font_size))  # Clamp between 12-24
                
                # Check if bar is too short for label - if so, place label above bar
                # Estimate if label will fit inside bar (rough heuristic based on log scale)
                if bar_top < 5.0:  # Short bars (less than $5 in log scale)
                    # Place label above the bar
                    label_y = bar_top * 1.08
                    va_align = 'bottom'
                else:
                    # Place label in the middle of the bar
                    label_y = bar_top * 0.7  # Position at 70% of bar height
                    va_align = 'center'
                
                plt.text(bar.get_x() + bar.get_width()/2., label_y,
                        f'${original_value:.2f}', ha='center', va=va_align, 
                        fontsize=font_size, rotation=90, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='black', linewidth=0.3))
    
    plt.xlabel('Categories', fontsize=34, fontweight='bold')
    plt.ylabel('Mean Absolute Error ($) - Log Scale', fontsize=34, fontweight='bold')
    plt.xticks(x + width * (len(models) - 1) / 2, categories, rotation=45, ha='right', fontsize=30)
    
    # Simple log scale with clean major ticks only
    from matplotlib.ticker import LogLocator, LogFormatter
    plt.gca().yaxis.set_major_locator(LogLocator(base=10, numticks=10))
    plt.tick_params(axis='y', which='major', labelsize=28)  # Make y-axis tick labels bigger
    
    plt.legend(loc='upper right', fontsize=18)  # Move legend inside the graph
    plt.grid(True, axis='y', alpha=0.3, which='both')
    
    # Add some extra space at top for labels that might be above bars
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.15)  # Add 15% more space at the top
    
    # Move title higher up with more padding
    plt.title(title, fontsize=36, fontweight='bold', pad=50)
    
    # Ensure y-axis shows proper log scale formatting
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created plot: {filename} with {len(categories)} categories and {len(models)} models")

def main():
    print("Loading diagnostic data from MLP models...")
    data = load_all_diagnostic_data()
    
    if not data:
        print("No diagnostic data found!")
        return
    
    print(f"\nLoaded {len(data)} models:")
    for model, analyses in data.items():
        print(f"  - {model}: {list(analyses.keys())}")
    
    # Create comparison plots for each analysis type
    analysis_configs = [
        ('price_quantiles', 'Mean Absolute Error by Price Quantile', 'mae_by_price_quantile.png'),
        ('moneyness', 'Mean Absolute Error by Moneyness', 'mae_by_moneyness.png'),
        ('time_to_maturity', 'Mean Absolute Error by Time to Maturity', 'mae_by_ttm.png'),
        ('historical_volatility', 'Mean Absolute Error by Historical Volatility', 'mae_by_hist_vol.png'),
        ('volume', 'Mean Absolute Error by Volume', 'mae_by_volume.png'),
        ('equity_uncertainty', 'Mean Absolute Error by Equity Uncertainty (EPU)', 'mae_by_epu.png'),
    ]
    
    for analysis_type, title, filename in analysis_configs:
        create_comparison_plot(data, analysis_type, title, filename)
    
    # Load individual prediction data
    frames = load_individual_predictions()
    
    # Create scatter plots
    plot_predictions_vs_true(frames)
    plot_residuals_vs_true(frames)
    
    # Save summary data
    summary = {
        'models_loaded': list(data.keys()),
        'analysis_types': list(next(iter(data.values())).keys()) if data else [],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, 'diagnostic_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDiagnostic comparison plots written to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
