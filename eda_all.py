"""
EDA Script for Options Data Analysis
- Loads raw options data and final processed dataset
- Creates numbered and titled plots for comprehensive analysis
- Handles missing data robustly
- Saves all figures to output directory
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Disable interactive plotting
plt.ioff()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, "00Data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_all_figures")

figure_counter = 1

def create_plot(title, filename, plot_func, data, figsize=(10, 6)):
    # Creates, titles, and saves a plot with proper numbering and error handling.
    
    global figure_counter
    try:
        plt.figure(figsize=figsize)
        plot_func(data)
        
        full_title = f"Figure {figure_counter}: {title}"
        plt.suptitle(full_title, fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        numbered_filename = f"{figure_counter:02d}_{filename}"
        plt.savefig(os.path.join(OUTPUT_DIR, numbered_filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Successfully created: {full_title}")
        figure_counter += 1
        
    except Exception as e:
        print(f"[ERROR] Failed to create plot '{title}': {e}")
        plt.close()

def export_data_range_diagnostics(df, target_column='mid_price', output_dir=OUTPUT_DIR):

    try:
        diagnostics_file = os.path.join(output_dir, "data_range_diagnostics.txt")
        
        with open(diagnostics_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DATA RANGE DIAGNOSTICS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic dataset info
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}\n")
            f.write(f"Target Column: {target_column}\n\n")
            
            # Price statistics
            f.write("PRICE STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Min Price: ${df[target_column].min():.2f}\n")
            f.write(f"Max Price: ${df[target_column].max():.2f}\n")
            f.write(f"Mean Price: ${df[target_column].mean():.2f}\n")
            f.write(f"Median Price: ${df[target_column].median():.2f}\n")
            f.write(f"Standard Deviation: ${df[target_column].std():.2f}\n\n")
            
            # Percentiles
            f.write("PRICE PERCENTILES\n")
            f.write("-" * 40 + "\n")
            percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                value = df[target_column].quantile(p/100)
                f.write(f"{p}th percentile: ${value:.2f}\n")
            f.write("\n")
            
            # Price range distribution
            f.write("PRICE RANGE DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            price_ranges = [0, 10, 50, 100, 200, 500, 1000, float('inf')]
            range_labels = ['<$10', '$10-50', '$50-100', '$100-200', '$200-500', '$500-1000', '>$1000']
            
            for i, label in enumerate(range_labels):
                if i < len(price_ranges) - 1:
                    count = ((df[target_column] >= price_ranges[i]) & (df[target_column] < price_ranges[i+1])).sum()
                    pct = count / len(df) * 100
                    f.write(f"{label}: {count:,} options ({pct:.1f}%)\n")
            f.write("\n")
            
            # Additional features if available
            if 'strike_price' in df.columns:
                f.write("STRIKE PRICE STATISTICS PER 100 OPTIONS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Strike Price Range: ${df['strike_price'].min():.2f} - ${df['strike_price'].max():.2f}\n")
                f.write(f"Strike Price Mean: ${df['strike_price'].mean():.2f}\n")
                f.write(f"Strike Price Median: ${df['strike_price'].median():.2f}\n\n")
            
            if 'spx_close' in df.columns:
                f.write("UNDERLYING PRICE STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"SPX Close Range: ${df['spx_close'].min():.2f} - ${df['spx_close'].max():.2f}\n")
                f.write(f"SPX Close Mean: ${df['spx_close'].mean():.2f}\n")
                f.write(f"SPX Close Median: ${df['spx_close'].median():.2f}\n\n")
            
            if 'days_to_maturity' in df.columns:
                f.write("TIME TO MATURITY STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Days to Maturity Range: {df['days_to_maturity'].min()} - {df['days_to_maturity'].max()} days\n")
                f.write(f"Days to Maturity Mean: {df['days_to_maturity'].mean():.1f} days\n")
                f.write(f"Days to Maturity Median: {df['days_to_maturity'].median():.1f} days\n\n")
            
            if 'volume' in df.columns:
                f.write("VOLUME STATISTICS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Volume Range: {df['volume'].min()} - {df['volume'].max()}\n")
                f.write(f"Volume Mean: {df['volume'].mean():.1f}\n")
                f.write(f"Volume Median: {df['volume'].median():.1f}\n")
                zero_volume_count = (df['volume'] == 0).sum()
                zero_volume_pct = zero_volume_count / len(df) * 100
                f.write(f"Zero Volume Options: {zero_volume_count:,} ({zero_volume_pct:.1f}%)\n\n")

            # Bid-Ask spread quality (if bids/offers present)
            if {'best_bid', 'best_offer'}.issubset(df.columns):
                f.write("BID-ASK SPREAD QUALITY\n")
                f.write("-" * 40 + "\n")
                spread = (df['best_offer'] - df['best_bid']).replace([np.inf, -np.inf], np.nan).dropna()
                if len(spread) > 0:
                    f.write(f"Spread Range: ${spread.min():.2f} - ${spread.max():.2f}\n")
                    f.write(f"Spread Mean: ${spread.mean():.2f}\n")
                    f.write(f"Spread Median: ${spread.median():.2f}\n")
                    neg_spread = (df['best_offer'] < df['best_bid']).sum()
                    neg_pct = neg_spread / len(df) * 100
                    zero_spread = (spread == 0).sum()
                    f.write(f"Negative Spreads: {neg_spread:,} ({neg_pct:.3f}%)\n")
                    f.write(f"Zero Spreads: {zero_spread:,}\n\n")
                else:
                    f.write("No valid spread data available.\n\n")

            # Open interest summary
            if 'open_interest' in df.columns:
                f.write("OPEN INTEREST STATISTICS\n")
                f.write("-" * 40 + "\n")
                oi = df['open_interest']
                f.write(f"Open Interest Range: {oi.min()} - {oi.max()}\n")
                f.write(f"Open Interest Mean: {oi.mean():.1f}\n")
                f.write(f"Open Interest Median: {oi.median():.1f}\n")
                zero_oi = (oi == 0).sum()
                f.write(f"Zero Open Interest: {zero_oi:,} ({zero_oi/len(df)*100:.1f}%)\n\n")

            # Calls vs Puts distribution
            if 'cp_flag' in df.columns:
                f.write("CALLS VS PUTS DISTRIBUTION\n")
                f.write("-" * 40 + "\n")
                cp_counts = df['cp_flag'].value_counts(dropna=False)
                total = len(df)
                for k, v in cp_counts.items():
                    label = 'Calls' if k == 'C' else ('Puts' if k == 'P' else 'Unknown')
                    f.write(f"{label}: {v:,} ({v/total*100:.1f}%)\n")
                f.write("\n")

            # Moneyness (on raw if available via spx_close & strike_price)
            if {'spx_close', 'strike_price'}.issubset(df.columns):
                with np.errstate(divide='ignore', invalid='ignore'):
                    mny = (df['spx_close'] / df['strike_price']).replace([np.inf, -np.inf], np.nan)
                mny = mny.dropna()
                if len(mny) > 0:
                    f.write("MONEYNESS (S/K) STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Range: {mny.min():.4f} - {mny.max():.4f}\n")
                    f.write(f"Mean: {mny.mean():.4f}\n")
                    f.write(f"Median: {mny.median():.4f}\n")
                    # Percentiles
                    for p in [1,5,25,50,75,95,99]:
                        f.write(f"P{p}: {mny.quantile(p/100):.4f}\n")
                    # Buckets
                    bins = [0, 0.8, 0.95, 1.05, 1.2, np.inf]
                    labels = ['Deep OTM', 'OTM', 'ATM', 'ITM', 'Deep ITM']
                    cats = pd.cut(mny, bins=bins, labels=labels)
                    f.write("Bucket Distribution:\n")
                    for lab in labels:
                        cnt = (cats == lab).sum()
                        f.write(f"  {lab}: {cnt:,} ({cnt/len(mny)*100:.1f}%)\n")
                    f.write("\n")

            # Implied and historical volatility summaries
            if 'impl_volatility' in df.columns:
                iv = df['impl_volatility'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(iv) > 0:
                    f.write("IMPLIED VOLATILITY STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Range: {iv.min():.4f} - {iv.max():.4f}\n")
                    f.write(f"Mean: {iv.mean():.4f}\n")
                    f.write(f"Median: {iv.median():.4f}\n")
                    f.write(f"P5-P95: {iv.quantile(0.05):.4f} - {iv.quantile(0.95):.4f}\n")
                    invalid = ((df['impl_volatility'] <= 0) | (df['impl_volatility'] > 5)).sum()
                    f.write(f"Invalid IV values (<=0 or >5): {invalid:,}\n\n")

            if 'historical_volatility' in df.columns:
                hv = df['historical_volatility'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(hv) > 0:
                    f.write("HISTORICAL VOLATILITY STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Range: {hv.min():.4f} - {hv.max():.4f}\n")
                    f.write(f"Mean: {hv.mean():.4f}\n")
                    f.write(f"Median: {hv.median():.4f}\n\n")

            # Rates and dividends
            if 'risk_free_rate' in df.columns:
                rfr = df['risk_free_rate'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(rfr) > 0:
                    f.write("RISK-FREE RATE (cont.) STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Range: {rfr.min():.5f} - {rfr.max():.5f}\n")
                    f.write(f"Mean: {rfr.mean():.5f}\n")
                    f.write(f"Median: {rfr.median():.5f}\n\n")

            if 'dividend_rate' in df.columns:
                dr = df['dividend_rate'].replace([np.inf, -np.inf], np.nan).dropna()
                if len(dr) > 0:
                    f.write("DIVIDEND RATE STATISTICS\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Range: {dr.min():.5f} - {dr.max():.5f}\n")
                    f.write(f"Mean: {dr.mean():.5f}\n")
                    f.write(f"Median: {dr.median():.5f}\n\n")

            # Economic indices
            econ_cols = [('epu_index', 'EPU Index'), ('equity_uncertainty', 'Equity Uncertainty'), ('equity_volatility', 'Equity Volatility')]
            avail_econ = [c for c, _ in econ_cols if c in df.columns]
            if avail_econ:
                f.write("ECONOMIC INDICES SUMMARY\n")
                f.write("-" * 40 + "\n")
                for c, label in econ_cols:
                    if c in df.columns:
                        vals = df[c].replace([np.inf, -np.inf], np.nan).dropna()
                        if len(vals) == 0:
                            continue
                        f.write(f"{label}: mean={vals.mean():.2f}, median={vals.median():.2f}, p5={vals.quantile(0.05):.2f}, p95={vals.quantile(0.95):.2f}\n")
                f.write("\n")

            # Days to maturity buckets
            if 'days_to_maturity' in df.columns:
                f.write("DAYS TO MATURITY BUCKETS\n")
                f.write("-" * 40 + "\n")
                buckets = [(0,7,'<1W'), (8,30,'1W-1M'), (31,90,'1-3M'), (91,365,'3M-1Y'), (366, 2000, '>1Y')]
                for a,b,label in buckets:
                    cnt = ((df['days_to_maturity'] >= a) & (df['days_to_maturity'] <= b)).sum()
                    f.write(f"{label}: {cnt:,} ({cnt/len(df)*100:.1f}%)\n")
                f.write("\n")

            # Volume buckets (activity)
            if 'volume' in df.columns:
                f.write("VOLUME BUCKETS (ACTIVITY)\n")
                f.write("-" * 40 + "\n")
                vb = [(1,10,'1-10'), (11,100,'11-100'), (101,1000,'101-1k'), (1001,10000,'1k-10k'), (10001, np.inf, '>10k')]
                for a,b,label in vb:
                    cnt = ((df['volume'] >= a) & (df['volume'] <= b)).sum()
                    f.write(f"{label}: {cnt:,} ({cnt/len(df)*100:.1f}%)\n")
                f.write("\n")

            # Missingness overview for key columns
            key_cols = ['mid_price','best_bid','best_offer','strike_price','spx_close','days_to_maturity','volume','impl_volatility','historical_volatility','risk_free_rate','dividend_rate','open_interest','cp_flag']
            present = [c for c in key_cols if c in df.columns]
            if present:
                f.write("MISSINGNESS SUMMARY (KEY COLUMNS)\n")
                f.write("-" * 40 + "\n")
                for c in present:
                    miss = df[c].isna().sum()
                    f.write(f"{c}: missing {miss:,} ({miss/len(df)*100:.2f}%)\n")
                f.write("\n")

            # Date coverage
            if 'date' in df.columns:
                f.write("DATE COVERAGE\n")
                f.write("-" * 40 + "\n")
                n_dates = df['date'].nunique()
                f.write(f"Unique Trading Dates: {n_dates:,}\n")
                f.write(f"Avg options per date: {len(df)/max(n_dates,1):.0f}\n\n")

            f.write("=" * 60 + "\n")
            f.write("End of Data Range Diagnostics Report\n")
            f.write("=" * 60 + "\n")
        
        print(f"Data Range Diagnostics exported to: {diagnostics_file}")
        
    except Exception as e:
        print(f"[ERROR] Failed to export data range diagnostics: {e}")

def main():
    """Main function to run the EDA script."""
    
    # Ensure non-interactive mode
    plt.ioff()
    
    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print('--- STARTING EDA ---')
    
    options = None
    
    try:
        raw_file_path = None
        if os.path.exists(DATA_PATH):
            for f in os.listdir(DATA_PATH):
                if f.lower() == 'data1.csv':
                    raw_file_path = os.path.join(DATA_PATH, f)
                    break
        
        if raw_file_path is None:
            raise FileNotFoundError("No file matching 'data1.csv' found")
        
        options = pd.read_csv(raw_file_path)
        options['date'] = pd.to_datetime(options['date'])
        options['exdate'] = pd.to_datetime(options['exdate'])
        print("Raw options data loaded successfully")
        
        # Calculate missing columns if needed
        if 'days_to_maturity' not in options.columns:
            options['days_to_maturity'] = (options['exdate'] - options['date']).dt.days
            print("Calculated days_to_maturity")
            
        if 'mid_price' not in options.columns and 'best_bid' in options.columns and 'best_offer' in options.columns:
            options['mid_price'] = (options['best_bid'] + options['best_offer']) / 2
            print("Calculated mid_price")
        
        # Export data range diagnostics
        print("\n=== EXPORTING DATA RANGE DIAGNOSTICS ===")
        export_data_range_diagnostics(options, target_column='mid_price')
    
    except Exception as e:
        print(f"[ERROR] Could not load raw data: {e}")
        print("Skipping raw data plots...")
    
    # --- 2. RAW DATA PLOTS ---
    if options is not None:
        print("\n=== CREATING RAW DATA PLOTS ===")
        
        # Plot 1: 3D Implied Volatility Surface
        def plot_iv_surface(data):
            required_cols = ['strike_price', 'days_to_maturity', 'impl_volatility']
            if not all(col in data.columns for col in required_cols):
                print("[WARN] Missing columns for IV surface plot")
                return
            
            plot_data = data[required_cols].dropna()
            plot_data = plot_data[
                (plot_data['impl_volatility'] > 0) & 
                (plot_data['impl_volatility'] < 4) &
                (plot_data['days_to_maturity'] > 0)
            ]
            
            if len(plot_data) < 10:
                print("[WARN] Insufficient data for IV surface")
                return
            
            fig = plt.gcf()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(
                plot_data['strike_price'], 
                plot_data['days_to_maturity'], 
                plot_data['impl_volatility'], 
                cmap='viridis', 
                alpha=0.8
            )
            ax.set_xlabel('Strike Price')
            ax.set_ylabel('Days to Maturity')
            ax.set_zlabel('Implied Volatility')
        
        create_plot('3D Implied Volatility Surface (Raw)', 'iv_surface_raw.png', plot_iv_surface, options, figsize=(12, 8))
        
        # Plot 2: Price vs Implied Volatility by Days to Maturity
        def plot_price_iv_by_dtm(data):
            required_cols = ['mid_price', 'impl_volatility', 'days_to_maturity']
            if not all(col in data.columns for col in required_cols):
                print("[WARN] Missing columns for Price vs IV plot")
                return
            
            plot_data = data[required_cols].dropna()
            plot_data = plot_data[
                (plot_data['impl_volatility'] > 0) & 
                (plot_data['impl_volatility'] < 4) &
                (plot_data['mid_price'] > 0)
            ]
            
            if plot_data.empty:
                print("[WARN] No valid data for Price vs IV plot")
                return
            
            # Define DTM ranges
            dtm_ranges = [
                (0, 30, '0-30 days'),
                (31, 60, '31-60 days'),
                (61, 90, '61-90 days'),
                (91, 120, '91-120 days'),
                (121, 180, '121-180 days'),
                (181, 270, '181-270 days'),
                (271, 365, '271-365 days'),
                (366, 1000, '365+ days')
            ]
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, (min_dtm, max_dtm, label) in enumerate(dtm_ranges):
                subset = plot_data[
                    (plot_data['days_to_maturity'] >= min_dtm) & 
                    (plot_data['days_to_maturity'] <= max_dtm)
                ]
                
                if len(subset) > 0:
                    scatter = axes[i].scatter(
                        subset['impl_volatility'], 
                        subset['mid_price'], 
                        c=subset['days_to_maturity'], 
                        cmap='viridis', 
                        alpha=0.7, 
                        s=15
                    )
                    axes[i].set_title(f'{label} (n={len(subset)})')
                    axes[i].set_xlabel('Implied Volatility')
                    axes[i].set_ylabel('Mid Price')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add colorbar for each subplot
                    cbar = plt.colorbar(scatter, ax=axes[i])
                    cbar.set_label('Days to Maturity', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
                else:
                    axes[i].set_title(f'{label} (No Data)')
                    axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
            
            plt.tight_layout()
        
        create_plot('Price vs. Implied Volatility by Days to Maturity (Raw)', 'price_vs_iv_by_dtm_raw.png', plot_price_iv_by_dtm, options, figsize=(20, 10))
        
        # Plot 3: Strike Price vs Implied Volatility by Days to Maturity
        def plot_strike_vs_iv_by_maturity(data):
            required_cols = ['strike_price', 'impl_volatility', 'days_to_maturity']
            if not all(col in data.columns for col in required_cols):
                print("[WARN] Missing columns for Strike vs IV plot")
                return
            
            plot_data = data[required_cols].dropna()
            plot_data = plot_data[
                (plot_data['impl_volatility'] > 0) & 
                (plot_data['impl_volatility'] < 2) &
                (plot_data['strike_price'] > 0) &
                (plot_data['days_to_maturity'] > 0)
            ]
            
            if plot_data.empty:
                print("[WARN] No valid data for Strike vs IV plot")
                return
            
            # Define DTM ranges with different color schemes for each range
            dtm_ranges = [
                (0, 30, '0-30 days', 'Reds'),
                (31, 60, '31-60 days', 'Oranges'),
                (61, 90, '61-90 days', 'YlOrBr'),
                (91, 120, '91-120 days', 'YlGn'),
                (121, 180, '121-180 days', 'Greens'),
                (181, 270, '181-270 days', 'Blues'),
                (271, 365, '271-365 days', 'Purples'),
                (366, 1000, '365+ days', 'plasma')
            ]
            
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for i, (min_dtm, max_dtm, label, colormap) in enumerate(dtm_ranges):
                subset = plot_data[
                    (plot_data['days_to_maturity'] >= min_dtm) & 
                    (plot_data['days_to_maturity'] <= max_dtm)
                ]
                
                if len(subset) > 0:
                    # Use specific color range for this maturity bucket
                    scatter = axes[i].scatter(
                        subset['strike_price'], 
                        subset['impl_volatility'], 
                        c=subset['days_to_maturity'], 
                        cmap=colormap, 
                        alpha=0.7, 
                        s=15,
                        vmin=min_dtm,  # Set color scale to the specific range
                        vmax=max_dtm if max_dtm < 1000 else subset['days_to_maturity'].max()
                    )
                    axes[i].set_title(f'{label} (n={len(subset):,})', fontsize=10, fontweight='bold')
                    axes[i].set_xlabel('Strike Price', fontsize=9)
                    axes[i].set_ylabel('Implied Volatility', fontsize=9)
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(labelsize=8)
                    
                    # Add colorbar for each subplot with appropriate range
                    cbar = plt.colorbar(scatter, ax=axes[i], shrink=0.8)
                    cbar.set_label('Days to Maturity', fontsize=8)
                    cbar.ax.tick_params(labelsize=7)
                    
                    # Set axis limits for better visualization
                    axes[i].set_xlim(subset['strike_price'].quantile(0.01), subset['strike_price'].quantile(0.99))
                    axes[i].set_ylim(0, subset['impl_volatility'].quantile(0.95))
                    
                else:
                    axes[i].set_title(f'{label} (No Data)', fontsize=10)
                    axes[i].text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                               transform=axes[i].transAxes, fontsize=12, color='gray')
                    axes[i].set_xlabel('Strike Price', fontsize=9)
                    axes[i].set_ylabel('Implied Volatility', fontsize=9)
            
            plt.tight_layout(pad=2.0)
        
        create_plot('Strike Price vs. Implied Volatility by Days to Maturity (Raw)', 'strike_vs_iv_by_dtm_raw.png', plot_strike_vs_iv_by_maturity, options, figsize=(20, 10))
    
    # --- 3. FINAL DATASET LOADING ---
    print("\n=== LOADING FINAL DATASET ===")
    try:
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "final_options_dataset.csv"))
        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])
        print("Final dataset loaded successfully")
    except FileNotFoundError:
        print("[ERROR] 'final_options_dataset.csv' not found. Cannot proceed with final plots.")
        return
    except Exception as e:
        print(f"[ERROR] Could not load final dataset: {e}")
        return
    
    # --- 4. FINAL DATASET PLOTS ---
    print("\n=== CREATING FINAL DATASET PLOTS ===")
    
    # Plot 4: Strike Price Distribution
    def plot_strike_dist(data):
        sns.histplot(data['strike_price'], bins=50, color='skyblue')
        plt.xlabel('Strike Price')
        plt.ylabel('Frequency')
    
    create_plot('Strike Price Distribution', 'strike_dist.png', plot_strike_dist, df)
    
    # Plot 5: Days to Maturity Distribution
    def plot_dtm_dist(data):
        sns.histplot(data['days_to_maturity'], bins=50, color='salmon')
        plt.xlabel('Days to Maturity')
        plt.ylabel('Frequency')
    
    create_plot('Days to Maturity Distribution', 'dtm_dist.png', plot_dtm_dist, df)
    
    # Plot 6: Mid Price Distribution
    def plot_mid_price_dist(data):
        sns.histplot(data['mid_price'], bins=50, color='lightgreen')
        plt.xlabel('Mid Price')
        plt.ylabel('Frequency')
    
    create_plot('Mid Price Distribution', 'mid_price_dist.png', plot_mid_price_dist, df)
    
    # Plot 7: Volume Distribution (Log Scale)
    def plot_volume_dist(data):
        sns.histplot(data['volume'], bins=50, color='orchid', log_scale=True)
        plt.xlabel('Volume (Log Scale)')
        plt.ylabel('Frequency')
    
    create_plot('Volume Distribution (Log Scale)', 'volume_dist.png', plot_volume_dist, df)
    
    # Plot 8: Historical Volatility Distribution
    def plot_hist_vol_dist(data):
        sns.histplot(data['historical_volatility'], bins=50, color='gold')
        plt.xlabel('Historical Volatility')
        plt.ylabel('Frequency')
    
    create_plot('Historical Volatility Distribution', 'hist_vol_dist.png', plot_hist_vol_dist, df)
    
    # Plot 9: Moneyness Distribution
    def plot_moneyness_dist(data):
        sns.histplot(data['moneyness'], bins=50, color='orange')
        plt.xlabel('Moneyness (S/K)')
        plt.ylabel('Frequency')
    
    create_plot('Moneyness Distribution (S/K)', 'moneyness_dist.png', plot_moneyness_dist, df)
    
    # Plot 10: Bid-Ask Spread Distribution
    def plot_bid_ask_dist(data):
        spread = data['best_offer'] - data['best_bid']
        spread = spread[(spread >= 0) & (spread <= 10)].dropna()
        sns.histplot(spread, bins=50, color='gold')
        plt.xlabel('Bid-Ask Spread')
        plt.ylabel('Frequency')
    
    create_plot('Bid-Ask Spread Distribution', 'bid_ask_dist.png', plot_bid_ask_dist, df)
    
    # Plot 11: Feature Correlation Heatmap
    def plot_correlation_heatmap(data):
        corr = data.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', annot_kws={"size": 8})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    create_plot('Feature Correlation Heatmap', 'corr_heatmap.png', plot_correlation_heatmap, df, figsize=(12, 10))
    
    # Plot 12: Volume vs Days to Maturity
    def plot_volume_vs_dtm(data):
        sns.scatterplot(x='days_to_maturity', y='volume', data=data, alpha=0.3)
        plt.xlabel('Days to Maturity')
        plt.ylabel('Volume')
    
    create_plot('Volume vs. Days to Maturity', 'vol_vs_dtm.png', plot_volume_vs_dtm, df)
    
    # Plot 13: Time Series of Key Variables
    def plot_time_series_1y(data):
        # Filter data to get closest to 1Y (365 days) for historical volatility
        
        # For historical volatility: get data closest to 365 days
        if 'days_to_maturity' in data.columns:
            hist_vol_data = data.copy()
            hist_vol_data['days_diff'] = abs(hist_vol_data['days_to_maturity'] - 365)
            hist_vol_filtered = hist_vol_data.loc[hist_vol_data.groupby('date')['days_diff'].idxmin()]
        else:
            hist_vol_filtered = data
        
        # For risk-free rate: get data in 340-380 days range
        if 'days_to_maturity' in data.columns:
            rf_data = data[
                (data['days_to_maturity'] >= 340) & 
                (data['days_to_maturity'] <= 380)
            ]
            if rf_data.empty:
                rf_data = data  # fallback to all data if no data in range
        else:
            rf_data = data
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # SPX Close Price
        if 'spx_close' in data.columns:
            axes[0].plot(data['date'], data['spx_close'], label='SPX Close', color='blue')
            axes[0].set_title('SPX Close Price Over Time')
            axes[0].set_ylabel('Price')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Historical Volatility (1Y)
        if not hist_vol_filtered.empty and 'historical_volatility' in hist_vol_filtered.columns:
            axes[1].plot(hist_vol_filtered['date'], hist_vol_filtered['historical_volatility'], 
                        label='Historical Volatility (1Y)', color='green', marker='o', markersize=2)
            axes[1].set_title('Historical Volatility Over Time (1Y Data)')
            axes[1].set_ylabel('Volatility')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # Risk-Free Rate (340-380 days)
        if not rf_data.empty and 'risk_free_rate' in rf_data.columns:
            axes[2].plot(rf_data['date'], rf_data['risk_free_rate'], 
                        label='Continuous Risk-Free Rate (1Y)', color='red', marker='o', markersize=2)
            axes[2].set_title('Continuous Risk-Free Rate (1Y)')
            axes[2].set_xlabel('Date')
            axes[2].set_ylabel('Rate')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    create_plot('Time Series of Key Financial Indicators (1Y)', 'time_series_1y.png', plot_time_series_1y, df, figsize=(14, 12))
    
    # Plot 14: Calls vs Puts Distribution
    def plot_calls_vs_puts(data):
        if 'cp_flag' not in data.columns:
            print("[WARN] Missing 'cp_flag' column for calls vs puts distribution")
            return
        
        cp_counts = data['cp_flag'].value_counts()
        colors = ['skyblue', 'lightcoral']
        labels = ['Calls' if x == 'C' else 'Puts' for x in cp_counts.index]
        
        plt.pie(cp_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.axis('equal')
    
    create_plot('Calls vs Puts Distribution', 'calls_vs_puts.png', plot_calls_vs_puts, df)
    
    # Plot 15: Dividend Rate Distribution
    def plot_dividend_rate_dist(data):
        if 'dividend_rate' not in data.columns:
            print("[WARN] Missing 'dividend_rate' column for dividend rate distribution")
            return
        
        dividend_data = data['dividend_rate'].dropna()
        if dividend_data.empty:
            print("[WARN] No valid dividend rate data")
            return
            
        sns.histplot(dividend_data, bins=50, color='green', alpha=0.7)
        plt.xlabel('Dividend Rate')
        plt.ylabel('Frequency')
    
    create_plot('Dividend Rate Distribution', 'dividend_rate_dist.png', plot_dividend_rate_dist, df)
    
    # Plot 16: Risk Free Rate Distribution
    def plot_risk_free_rate_dist(data):
        if 'risk_free_rate' not in data.columns:
            print("[WARN] Missing 'risk_free_rate' column for risk free rate distribution")
            return
        
        rfr_data = data['risk_free_rate'].dropna()
        if rfr_data.empty:
            print("[WARN] No valid risk free rate data")
            return
            
        sns.histplot(rfr_data, bins=50, color='purple', alpha=0.7)
        plt.xlabel('Risk Free Rate')
        plt.ylabel('Frequency')
    
    create_plot('Risk Free Rate Distribution', 'risk_free_rate_dist.png', plot_risk_free_rate_dist, df)
    
    # Plot 17: Time vs progression of EPU, Market Economic Uncertainty, Equity Market Volatility
    def plot_economic_indices_time_series(data):
        cols = {
            'epu_index': 'Economic Policy Uncertainty (EPU)',
            'equity_uncertainty': 'Market Economic Uncertainty',
            'equity_volatility': 'Equity Market Volatility'
        }
        present = [c for c in cols if c in data.columns]
        if not present:
            print("[WARN] No economic indices columns found for time series plot")
            return
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        axes = axes if isinstance(axes, np.ndarray) else [axes]
        for i, c in enumerate(['epu_index', 'equity_uncertainty', 'equity_volatility']):
            ax = axes[i]
            if c in data.columns:
                series = data[['date', c]].dropna().sort_values('date')
                if len(series) == 0:
                    ax.text(0.5, 0.5, f'No data for {cols[c]}', ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.plot(series['date'], series[c], color=['tab:blue','tab:orange','tab:green'][i], lw=1)
                    ax.set_title(cols[c])
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Missing column: {c}', ha='center', va='center', transform=ax.transAxes)
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
    
    create_plot('Economic Indices Over Time (EPU, Uncertainty, Equity Volatility)', 'economic_time_series.png', plot_economic_indices_time_series, df, figsize=(14, 12))
    
    # Plot 18: Distributions of EPU, Market Economic Uncertainty, Equity Market Volatility
    def plot_economic_indices_distributions(data):
        cols = [
            ('epu_index', 'EPU'),
            ('equity_uncertainty', 'Market Economic Uncertainty'),
            ('equity_volatility', 'Equity Market Volatility')
        ]
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        for i, (col, label) in enumerate(cols):
            ax = axes[i]
            if col in data.columns:
                vals = data[col].dropna()
                if len(vals) == 0:
                    ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center', transform=ax.transAxes)
                else:
                    sns.histplot(vals, bins=50, ax=ax, color=['tab:blue','tab:orange','tab:green'][i], alpha=0.7)
                    ax.set_xlabel(label)
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.2)
            else:
                ax.text(0.5, 0.5, f'Missing column: {col}', ha='center', va='center', transform=ax.transAxes)
        plt.tight_layout()
    
    create_plot('Economic Indices Distributions (EPU, Uncertainty, Equity Volatility)', 'economic_indices_distributions.png', plot_economic_indices_distributions, df, figsize=(12, 12))
    
    print('\n--- EDA COMPLETE ---')
    print(f'All figures saved to: {OUTPUT_DIR}')
    print(f'Total plots created: {figure_counter - 1}')

if __name__ == "__main__":
    main()
