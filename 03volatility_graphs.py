import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from pathlib import Path
import sys
from typing import List
 
# Suppress only specific warnings we know are safe
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class IVAnalyzer:
    """Complete implied volatility analysis with all redundancy removed"""
    
    def __init__(self, run_name: str = None, verbose: bool = True):
        self.run_name = run_name or self._get_latest_run_name()
        self.verbose = verbose
        self._setup_plotting_style()
        
    def _get_latest_run_name(self):
        """Return the latest run_YYYYMMDD_HHMMSS folder name (by mod time)"""
        base_results = self._base_input_results_dir()
        if not base_results.exists():
            return None
        run_dirs = [p for p in base_results.iterdir() if p.is_dir() and p.name.startswith('run_')]
        if not run_dirs:
            return None
        # Sort by modification time (newest first)
        run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return run_dirs[0].name

    def _setup_plotting_style(self):
        """Configure matplotlib style once"""
        plt.style.use('seaborn-v0_8')
        plt.rcParams.update({
            'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': 0.3,
            'axes.edgecolor': 'black', 'axes.linewidth': 0.5, 'figure.figsize': (12, 8), 'font.size': 12
        })

    @staticmethod
    def _ensure_parent(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def _results_dir(self):
        """Centralized *output* directory logic (now under vol_results/<run_name>)"""
        base = Path(__file__).parent / 'vol_results'
        run = self.run_name or 'unknown_run'
        out_dir = base / run
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @staticmethod
    def _base_input_results_dir() -> Path:
        return Path(__file__).parent / 'results'

    @classmethod
    def discover_run_names(cls) -> List[str]:
        base_results = cls._base_input_results_dir()
        if not base_results.exists():
            return []
        runs = [p.name for p in base_results.iterdir() if p.is_dir() and p.name.lower().startswith('run_mlp')]
        # include nested run folders too
        for sub in base_results.iterdir():
            if sub.is_dir() and not sub.name.lower().startswith('run_mlp'):
                for p in sub.rglob('run_MLP*'):
                    if p.is_dir():
                        runs.append(p.name)
        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for r in runs:
            if r not in seen:
                uniq.append(r)
                seen.add(r)
        return sorted(uniq)

    def _resolve_predictions_path(self) -> Path:
        """Robustly resolve the predictions CSV path.
        Priority:
        1) If self.run_name is set and results/<run_name>/mlp_predictions.csv exists, use it
        2) If self.run_name is set and results/<run_name> exists, try common alternate filenames
        3) Otherwise search recursively under results/ for the most recently modified mlp_predictions.csv
        4) As a last resort, search for any *.csv in results/** and pick the most recent
        Raises FileNotFoundError with a helpful hint if nothing is found.
        """
        base_results = self._base_input_results_dir()
        # 1) Direct match
        if self.run_name:
            run_dir = base_results / self.run_name
            direct = run_dir / 'mlp_predictions.csv'
            if direct.exists():
                return direct
            # 2) Alternate names inside the run dir
            if run_dir.exists():
                for alt_name in ['predictions.csv', 'mlp_preds.csv', 'preds.csv']:
                    candidate = run_dir / alt_name
                    if candidate.exists():
                        if self.verbose:
                            print(f"Found alternate predictions file: {candidate}")
                        return candidate
        # 3) Recursive search for mlp_predictions.csv
        mlp_hits = list(base_results.rglob('mlp_predictions.csv')) if base_results.exists() else []
        if mlp_hits:
            mlp_hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if self.verbose:
                print(f"Auto-selected latest mlp_predictions.csv: {mlp_hits[0]}")
            return mlp_hits[0]
        # 4) Fallback to any CSV
        any_csv = list(base_results.rglob('*.csv')) if base_results.exists() else []
        if any_csv:
            any_csv.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if self.verbose:
                print(f"WARNING: Using most recent CSV (name not standard): {any_csv[0]}")
            return any_csv[0]
        # Nothing found — build a helpful message
        runs = [p for p in base_results.iterdir() if p.is_dir()] if base_results.exists() else []
        run_list = "\n".join(f" - {p.name}" for p in sorted(runs)) if runs else "(no run folders found)"
        raise FileNotFoundError(
            f"No predictions CSV found. Looked under: {base_results}\n"
            f"Expected: results/<run_name>/mlp_predictions.csv\n"
            f"Current run_name: {self.run_name}\n"
            f"Available run folders:\n{run_list}"
        )

    @staticmethod
    def _black_scholes_call(S, K, T, r, sigma, q=0):
        """Black-Scholes call price (put removed as unused)"""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    def _calculate_vega(S, K, T, r, sigma, q=0):
        """Option vega calculation"""
        if T <= 0:
            return 0
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    @staticmethod
    def _arbitrage_bounds(S, K, T, r, q=0):
        """No-arbitrage bounds for call options"""
        disc_S, disc_K = S * np.exp(-q * T), K * np.exp(-r * T)
        return max(disc_S - disc_K, 0.0), disc_S

    def _implied_volatility(self, price, S, K, T, r, q=0):
        """Unified IV solver with bounds checking and fallbacks"""
        if T <= 0 or price <= 0:
            return np.nan
        
        lower, upper = self._arbitrage_bounds(S, K, T, r, q)
        if price < lower * 0.98 or price > upper * 1.02:
            return np.nan
        
        # Newton-Raphson
        sigma = 0.3
        for _ in range(50):
            try:
                bs_price = self._black_scholes_call(S, K, T, r, sigma, q)
                vega = self._calculate_vega(S, K, T, r, sigma, q)
                if abs(bs_price - price) < 1e-6:
                    return sigma
                if abs(vega) < 1e-10:
                    break
                sigma = max(0.001, min(sigma - (bs_price - price) / max(vega, 1e-12), 5.0))
            except:
                break
        
        # Bisection fallback
        vol_low, vol_high = 0.001, 5.0
        for _ in range(100):
            vol_mid = (vol_low + vol_high) / 2
            bs_price = self._black_scholes_call(S, K, T, r, vol_mid, q)
            if abs(bs_price - price) < 1e-6:
                return vol_mid if 0.001 <= vol_mid <= 5.0 else np.nan
            if bs_price > price:
                vol_high = vol_mid
            else:
                vol_low = vol_mid
        return np.nan

    def _standardize_columns(self, df):
        """Standardize time and add moneyness once"""
        # Time to expiry
        if 'time_to_expiry' not in df.columns:
            if 'days_to_maturity_input' in df.columns:
                df['time_to_expiry'] = df['days_to_maturity_input'] / 365.25
                df['days_to_maturity'] = df['days_to_maturity_input']  # For downstream use
            elif 'days_to_maturity' in df.columns:
                df['time_to_expiry'] = df['days_to_maturity'] / 365.25
            else:
                raise ValueError("Missing time column")
        
        # Moneyness (computed once)
        if 'moneyness' not in df.columns:
            with np.errstate(divide='ignore', invalid='ignore'):
                df['moneyness'] = df['spx_close_input'] / df['strike_price_input']
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure days_to_maturity exists for plotting
        if 'days_to_maturity' not in df.columns:
            df['days_to_maturity'] = (df['time_to_expiry'] * 365.25).round().astype(int)
        
        return df.dropna(subset=['moneyness'])

    def _describe_series(self, name: str, s: pd.Series):
        """Reusable descriptive statistics helper"""
        q = s.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        return {
            f'{name}_count': int(s.count()), f'{name}_mean': float(s.mean()), 
            f'{name}_std': float(s.std(ddof=1)), f'{name}_min': float(s.min()),
            f'{name}_p01': float(q.loc[0.01]), f'{name}_p05': float(q.loc[0.05]),
            f'{name}_q25': float(q.loc[0.25]), f'{name}_median': float(q.loc[0.5]),
            f'{name}_q75': float(q.loc[0.75]), f'{name}_p95': float(q.loc[0.95]),
            f'{name}_p99': float(q.loc[0.99]), f'{name}_max': float(s.max())
        }

    def _error_bar_legend(self):
        """Reusable error bar legend proxy"""
        return Line2D([0], [0], color='black', lw=1.5)

    def _harmonize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make column names consistent and backfill expected *_input/target columns from common aliases."""
        # Normalize header whitespace / BOM
        df = df.copy()
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)

        # Alias map: source_name -> destination_expected_name
        aliases = {
            # Targets
            'true_price': 'actual_mid_price',
            'actual_price': 'actual_mid_price',
            'y_true': 'actual_mid_price',
            'predicted_price': 'predicted_mid_price',
            'y_pred': 'predicted_mid_price',
            # Core BSM inputs (non-suffixed to *_input)
            'spx_close': 'spx_close_input',
            'strike_price': 'strike_price_input',
            'risk_free_rate': 'risk_free_rate_input',
            'dividend_rate': 'dividend_rate_input',
            'historical_volatility': 'historical_volatility_input',
            'days_to_maturity': 'days_to_maturity_input',
            # Sometimes different casing
            'Days_to_maturity': 'days_to_maturity_input',
            'Risk_free_rate': 'risk_free_rate_input',
            'Dividend_rate': 'dividend_rate_input',
        }

        # Create missing expected columns from aliases when possible
        for src, dst in aliases.items():
            if dst not in df.columns and src in df.columns:
                df[dst] = df[src]

        # If time info comes as time_in_years, create days_to_maturity_input and time_to_expiry
        if 'days_to_maturity_input' not in df.columns and 'time_in_years' in df.columns:
            df['time_to_expiry'] = df['time_in_years']
            df['days_to_maturity_input'] = (df['time_in_years'] * 365.25).round().astype(int)

        # Ensure time_to_expiry exists if days_to_maturity_input present
        if 'time_to_expiry' not in df.columns:
            if 'days_to_maturity_input' in df.columns:
                df['time_to_expiry'] = df['days_to_maturity_input'] / 365.25
            elif 'days_to_maturity' in df.columns:
                df['time_to_expiry'] = df['days_to_maturity'] / 365.25

        # Type coercion for numeric columns we rely on
        numeric_like = [
            'actual_mid_price','predicted_mid_price','spx_close_input','strike_price_input',
            'risk_free_rate_input','dividend_rate_input','historical_volatility_input',
            'days_to_maturity_input','time_to_expiry'
        ]
        for c in numeric_like:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        return df

    def load_and_preprocess_data(self):
        """Load and preprocess prediction data"""
        try:
            predictions_path = self._resolve_predictions_path()
            if self.verbose:
                print(f"Loading predictions from: {predictions_path}")
            df = pd.read_csv(predictions_path)
            if self.verbose:
                print(f"Successfully loaded {len(df)} rows from {predictions_path.name}")

            # Harmonize/alias columns and standardize core fields
            df = self._harmonize_columns(df)
            df = self._standardize_columns(df)

            # Validate required fields now that harmonization is done
            required = ['actual_mid_price', 'predicted_mid_price', 'spx_close_input',
                        'strike_price_input', 'risk_free_rate_input', 'time_to_expiry']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"Missing required columns AFTER harmonization: {missing}")
                return None
            return df
        except FileNotFoundError as e:
            print(str(e))
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None

    def calculate_iv_metrics(self, df):
        """Calculate IV metrics with clamping for stability"""
        if self.verbose:
            print(f"Processing {len(df):,} options for IV calculation")
        
        # Clamp predicted prices to no-arbitrage bounds for better IV solving
        df = df.copy()
        q = df['dividend_rate_input'].fillna(0) if 'dividend_rate_input' in df.columns else 0.0
        lower_bounds = np.maximum(
            df['spx_close_input'] * np.exp(-q * df['time_to_expiry']) - 
            df['strike_price_input'] * np.exp(-df['risk_free_rate_input'] * df['time_to_expiry']), 0
        )
        upper_bounds = df['spx_close_input'] * np.exp(-q * df['time_to_expiry'])
        
        df['predicted_mid_price_clamped'] = np.clip(
            df['predicted_mid_price'], 
            lower_bounds * 1.001, 
            upper_bounds * 0.999
        )
        
        # Calculate IVs
        df['actual_iv'] = df.apply(lambda row: self._implied_volatility(
            row['actual_mid_price'], row['spx_close_input'], row['strike_price_input'],
            row['time_to_expiry'], row['risk_free_rate_input'], row.get('dividend_rate_input', 0)
        ), axis=1)
        
        df['predicted_iv'] = df.apply(lambda row: self._implied_volatility(
            row['predicted_mid_price_clamped'], row['spx_close_input'], row['strike_price_input'],
            row['time_to_expiry'], row['risk_free_rate_input'], row.get('dividend_rate_input', 0)
        ), axis=1)
        
        # Clean and add error metrics
        initial_count = len(df)
        df_clean = df[(df['actual_iv'].notna()) & (df['predicted_iv'].notna()) & 
                     (df['actual_iv'] > 0.001) & (df['actual_iv'] < 5.0) &
                     (df['predicted_iv'] > 0.001) & (df['predicted_iv'] < 5.0)].copy()
        
        df_clean['iv_error'] = df_clean['predicted_iv'] - df_clean['actual_iv']
        df_clean['iv_error_bps'] = df_clean['iv_error'] * 10000
        df_clean['abs_iv_error'] = np.abs(df_clean['iv_error'])
        df_clean['iv_error_pct'] = (df_clean['iv_error'] / df_clean['actual_iv']) * 100
        
        # Add buckets for analysis
        df_clean['moneyness_bucket'] = pd.cut(df_clean['moneyness'], 
            bins=[0, 0.9, 0.95, 1.05, 1.1, np.inf], labels=['Deep OTM', 'OTM', 'ATM', 'ITM', 'Deep ITM'])
        df_clean['maturity_bucket'] = pd.cut(df_clean['days_to_maturity'],
            bins=[0, 7, 30, 90, 365, np.inf], labels=['<1W', '1W-1M', '1-3M', '3M-1Y', '>1Y'])
        
        # Store diagnostics
        success_rate = (len(df_clean) / initial_count) * 100
        df_clean.attrs['diagnostics'] = {
            'total_options': initial_count, 'success_count': len(df_clean),
            'success_rate_pct': success_rate, 'failed_count': initial_count - len(df_clean)
        }
        
        if self.verbose:
            print(f"IV calculation: {len(df_clean):,} successful ({success_rate:.1f}%)")
        
        return df_clean

    def plot_iv_analysis(self, df):
        """Generate IV analysis plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Implied Volatility Analysis', fontsize=16, fontweight='bold')
        
        # 1. IV Scatter
        axes[0,0].scatter(df['actual_iv'], df['predicted_iv'], alpha=0.3, s=1, c='blue')
        iv_range = [df['actual_iv'].min(), df['actual_iv'].max()]
        axes[0,0].plot(iv_range, iv_range, 'r--', alpha=0.8, linewidth=2, label='Perfect')
        z = np.polyfit(df['actual_iv'], df['predicted_iv'], 1)
        axes[0,0].plot(iv_range, np.poly1d(z)(iv_range), 'g-', alpha=0.7, linewidth=2, 
                      label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        r2 = np.corrcoef(df['actual_iv'], df['predicted_iv'])[0,1]**2
        axes[0,0].text(0.95, 0.05, f'R² = {r2:.4f}', transform=axes[0,0].transAxes, ha='right')
        axes[0,0].set_xlabel('Actual IV')
        axes[0,0].set_ylabel('Predicted IV')
        axes[0,0].set_title('Actual vs Predicted IV')
        axes[0,0].legend()
        
        # 2. Error Distribution (zoomed)
        p2_5, p97_5 = np.percentile(df['iv_error_bps'], [2.5, 97.5])
        center = np.median(df['iv_error_bps'])
        half_width = max(abs(p2_5 - center), abs(p97_5 - center)) * 1.2
        df_filt = df[(df['iv_error_bps'] >= center - half_width) & (df['iv_error_bps'] <= center + half_width)]
        
        axes[0,1].hist(df_filt['iv_error_bps'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0,1].axvline(np.mean(df_filt['iv_error_bps']), color='green', linestyle='-', linewidth=2,
                         label=f'Mean: {np.mean(df_filt["iv_error_bps"]):.1f} bps')
        axes[0,1].set_xlabel('IV Error (bps)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title(f'IV Error Distribution (±{half_width:.0f} bps)')
        axes[0,1].legend()
        
        # 3. Moneyness Analysis
        mon_stats = df.groupby('moneyness_bucket', observed=True)['iv_error_bps'].agg(['mean', 'std', 'count']).dropna()
        if not mon_stats.empty:
            x = np.arange(len(mon_stats))
            bars = axes[1,0].bar(x, mon_stats['mean'], yerr=mon_stats['std'], capsize=5, alpha=0.7, color='lightcoral')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(mon_stats.index, rotation=45, ha='right')
            axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            for bar, count, mean, std in zip(bars, mon_stats['count'], mon_stats['mean'], mon_stats['std']):
                # Add count on top
                axes[1,0].text(bar.get_x() + bar.get_width()/2, mean + std + max(abs(mon_stats['mean'])) * 0.02,
                              f'n={int(count):,}', ha='center', va='bottom', fontsize=9)
                # Add mean value on bar with contrast guard
                txt_color = 'black'
                axes[1,0].text(bar.get_x() + bar.get_width()/2, mean/2,
                              f'{mean:.1f}', ha='center', va='center', fontsize=11, fontweight='bold', color=txt_color)
            axes[1,0].legend([self._error_bar_legend()], ['±1 SD'])
            axes[1,0].set_xlabel('Moneyness')
            axes[1,0].set_ylabel('Mean IV Error (bps)')
            axes[1,0].set_title('IV Error by Moneyness')
        
        # 4. Maturity Analysis  
        mat_stats = df.groupby('maturity_bucket', observed=True)['iv_error_bps'].agg(['mean', 'std', 'count']).dropna()
        if not mat_stats.empty:
            x = np.arange(len(mat_stats))
            bars = axes[1,1].bar(x, mat_stats['mean'], yerr=mat_stats['std'], capsize=5, alpha=0.7, color='lightgreen')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(mat_stats.index, rotation=45, ha='right')
            axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            for bar, count, mean, std in zip(bars, mat_stats['count'], mat_stats['mean'], mat_stats['std']):
                # Add count on top
                axes[1,1].text(bar.get_x() + bar.get_width()/2, mean + std + max(abs(mat_stats['mean'])) * 0.02,
                              f'n={int(count):,}', ha='center', va='bottom', fontsize=9)
                # Add mean value on bar with contrast guard
                txt_color = 'black' if abs(mean) > 1e-3 else 'black'
                axes[1,1].text(bar.get_x() + bar.get_width()/2, mean/2,
                              f'{mean:.1f}', ha='center', va='center', fontsize=11, fontweight='bold', color=txt_color)
            axes[1,1].legend([self._error_bar_legend()], ['±1 SD'])
            axes[1,1].set_xlabel('Time to Maturity')
            axes[1,1].set_ylabel('Mean IV Error (bps)')
            axes[1,1].set_title('IV Error by Maturity')
        
        plt.tight_layout()
        output_path = self._results_dir() / 'iv_analysis.png'
        self._ensure_parent(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        if self.verbose:
            print(f"IV analysis plots saved to: {output_path}")

    def plot_volatility_smile_and_residuals(self, df):
        """Generate volatility smile and residual plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Volatility Smile and Residual Analysis', fontsize=16, fontweight='bold')
        
        atm_strike = float(df['spx_close_input'].median()) if not df['spx_close_input'].empty else 0.0
        
        # 1. Volatility Smile - All options
        scatter = axes[0,0].scatter(df['strike_price_input'], df['predicted_iv'], c=df['days_to_maturity'], cmap='turbo', s=1, alpha=0.5)
        plt.colorbar(scatter, ax=axes[0,0], label='Days to Maturity')
        axes[0,0].axvline(x=atm_strike, color='red', linestyle='--', alpha=0.5, label=f'ATM (~${atm_strike:.0f})')
        axes[0,0].set_xlabel('Strike Price ($)')
        axes[0,0].set_ylabel('Predicted IV')
        axes[0,0].set_title('Volatility Smile (All Options)')
        axes[0,0].legend()
        
        # 2. Short-term options (≤200 days)
        df_200 = df[df['days_to_maturity'] <= 200]
        if not df_200.empty:
            scatter2 = axes[0,1].scatter(df_200['strike_price_input'], df_200['predicted_iv'],
                                        c=df_200['days_to_maturity'], cmap='turbo', s=1, alpha=0.5)
            plt.colorbar(scatter2, ax=axes[0,1], label='Days to Maturity')
            axes[0,1].axvline(x=atm_strike, color='red', linestyle='--', alpha=0.5)
        axes[0,1].set_xlabel('Strike Price ($)')
        axes[0,1].set_ylabel('Predicted IV')
        axes[0,1].set_title('Volatility Smile (≤200 Days)')
        
        # 3. Residuals vs Actual IV
        axes[1,0].scatter(df['actual_iv'], df['iv_error'], alpha=0.3, s=1, c='blue')
        axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Guard polyfit on small samples
        if len(df) >= 2:
            z = np.polyfit(df['actual_iv'], df['iv_error'], 1)
            x_trend = np.linspace(df['actual_iv'].min(), df['actual_iv'].max(), 100)
            axes[1,0].plot(x_trend, np.poly1d(z)(x_trend), 'g-', linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.3f}')
            axes[1,0].legend()
        
        axes[1,0].set_xlabel('Actual IV')
        axes[1,0].set_ylabel('IV Error')
        axes[1,0].set_title('Residuals vs Actual IV')
        
        # 4. Term Structure
        term_stats = df.groupby('maturity_bucket', observed=True)[['actual_iv', 'predicted_iv']].agg(['mean', 'count']).dropna()
        if not term_stats.empty:
            x = np.arange(len(term_stats))
            width = 0.35
            actual_means = term_stats[('actual_iv', 'mean')]
            predicted_means = term_stats[('predicted_iv', 'mean')]
            bars1 = axes[1,1].bar(x - width/2, actual_means, width, label='Actual IV', color='navy', alpha=0.8)
            bars2 = axes[1,1].bar(x + width/2, predicted_means, width, label='Predicted IV', color='crimson', alpha=0.8)
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(term_stats.index, rotation=45, ha='right')
            axes[1,1].set_xlabel('Time to Maturity')
            axes[1,1].set_ylabel('Average IV')
            axes[1,1].set_title('Term Structure of IV (Actual vs Predicted)')
            axes[1,1].legend()
            # Add value labels
            for bar, mean in zip(bars1, actual_means):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, float(mean) + float(actual_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='navy')
            for bar, mean in zip(bars2, predicted_means):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, float(mean) + float(predicted_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color='crimson')
        
        plt.tight_layout()
        output_path = self._results_dir() / 'volatility_smile_analysis.png'
        self._ensure_parent(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    def plot_detailed_iv_breakdowns(self, df):
        """Generate detailed IV breakdowns by various dimensions"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Detailed IV Analysis Breakdowns', fontsize=16, fontweight='bold')
        
        # 1. Average Predicted IV by Moneyness
        mon_stats = df.groupby('moneyness_bucket', observed=True)['predicted_iv'].agg(['mean', 'std', 'count']).dropna()
        if not mon_stats.empty:
            x = np.arange(len(mon_stats))
            bars = axes[0,0].bar(x, mon_stats['mean'], yerr=mon_stats['std'], capsize=5, alpha=0.7, color='steelblue')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(mon_stats.index, rotation=45, ha='right')
            axes[0,0].set_ylabel('Average Predicted IV')
            axes[0,0].set_title('Predicted IV by Moneyness')
            for bar, count, mean, std in zip(bars, mon_stats['count'], mon_stats['mean'], mon_stats['std']):
                # Count on top
                axes[0,0].text(bar.get_x() + bar.get_width()/2, mean + std + max(mon_stats['mean']) * 0.02,
                              f'n={int(count):,}', ha='center', va='bottom', fontsize=9)
                # Mean value on bar with contrast guard
                txt_color = 'white' if mean > 1e-3 else 'black'
                axes[0,0].text(bar.get_x() + bar.get_width()/2, mean/2,
                              f'{mean:.3f}', ha='center', va='center', fontsize=9, fontweight='bold', color=txt_color)
        
        # 2. Average Actual IV by Moneyness
        mon_actual = df.groupby('moneyness_bucket', observed=True)['actual_iv'].agg(['mean', 'std', 'count']).dropna()
        if not mon_actual.empty:
            x = np.arange(len(mon_actual))
            bars = axes[0,1].bar(x, mon_actual['mean'], yerr=mon_actual['std'], capsize=5, alpha=0.7, color='darkseagreen')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(mon_actual.index, rotation=45, ha='right')
            axes[0,1].set_ylabel('Average Actual IV')
            axes[0,1].set_title('Actual IV by Moneyness')
            for bar, count, mean, std in zip(bars, mon_actual['count'], mon_actual['mean'], mon_actual['std']):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, mean + std + max(mon_actual['mean']) * 0.02,
                              f'n={int(count):,}', ha='center', va='bottom', fontsize=9)
                txt_color = 'white' if mean > 1e-3 else 'black'
                axes[0,1].text(bar.get_x() + bar.get_width()/2, mean/2,
                              f'{mean:.3f}', ha='center', va='center', fontsize=9, fontweight='bold', color=txt_color)
        
        # 3. IV by Strike Price Ranges (bar: Actual vs Predicted)
        df['strike_bucket'] = pd.cut(df['strike_price_input'], bins=10, precision=0)
        strike_stats_both = df.groupby('strike_bucket', observed=True)[['actual_iv','predicted_iv']].agg(['mean','count']).dropna()
        if not strike_stats_both.empty:
            x = np.arange(len(strike_stats_both))
            width = 0.35
            labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in strike_stats_both.index]
            actual_means = strike_stats_both[('actual_iv','mean')]
            predicted_means = strike_stats_both[('predicted_iv','mean')]
            bars1 = axes[0,2].bar(x - width/2, actual_means, width, label='Actual IV', color='navy', alpha=0.8)
            bars2 = axes[0,2].bar(x + width/2, predicted_means, width, label='Predicted IV', color='crimson', alpha=0.8)
            axes[0,2].set_xticks(x)
            axes[0,2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[0,2].set_ylabel('Average IV')
            axes[0,2].set_xlabel('Strike Price Range ($)')
            axes[0,2].set_title('Actual vs Predicted IV by Strike Price (Bar)')
            axes[0,2].legend()
            for bar, mean in zip(bars1, actual_means):
                axes[0,2].text(bar.get_x() + bar.get_width()/2, float(mean) + float(actual_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='navy')
            for bar, mean in zip(bars2, predicted_means):
                axes[0,2].text(bar.get_x() + bar.get_width()/2, float(mean) + float(predicted_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='crimson')
        
        # 4. IV by Underlying Price Ranges (bar: Actual vs Predicted)
        df['spx_bucket'] = pd.cut(df['spx_close_input'], bins=8, precision=0)
        spx_stats_both = df.groupby('spx_bucket', observed=True)[['actual_iv','predicted_iv']].agg(['mean','count']).dropna()
        if not spx_stats_both.empty:
            x = np.arange(len(spx_stats_both))
            width = 0.35
            labels = [f"{int(interval.left)}-{int(interval.right)}" for interval in spx_stats_both.index]
            actual_means = spx_stats_both[('actual_iv','mean')]
            predicted_means = spx_stats_both[('predicted_iv','mean')]
            bars1 = axes[1,0].bar(x - width/2, actual_means, width, label='Actual IV', color='navy', alpha=0.8)
            bars2 = axes[1,0].bar(x + width/2, predicted_means, width, label='Predicted IV', color='crimson', alpha=0.8)
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[1,0].set_ylabel('Average IV')
            axes[1,0].set_xlabel('SPX Price Range ($)')
            axes[1,0].set_title('Actual vs Predicted IV by SPX Price (Bar)')
            axes[1,0].legend()
            for bar, mean in zip(bars1, actual_means):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, float(mean) + float(actual_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='navy')
            for bar, mean in zip(bars2, predicted_means):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, float(mean) + float(predicted_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='crimson')
        
        # 5. Average IV by Risk-Free Rate Ranges (bar: Actual vs Predicted)
        df['rfr_bucket'] = pd.cut(df['risk_free_rate_input'], bins=6, precision=4)
        rfr_stats_both = df.groupby('rfr_bucket', observed=True)[['actual_iv','predicted_iv']].agg(['mean','count']).dropna()
        if not rfr_stats_both.empty:
            x = np.arange(len(rfr_stats_both))
            width = 0.35
            labels = [f"{interval.left:.3f}-{interval.right:.3f}" for interval in rfr_stats_both.index]
            actual_means = rfr_stats_both[('actual_iv','mean')]
            predicted_means = rfr_stats_both[('predicted_iv','mean')]
            bars1 = axes[1,1].bar(x - width/2, actual_means, width, label='Actual IV', color='navy', alpha=0.8)
            bars2 = axes[1,1].bar(x + width/2, predicted_means, width, label='Predicted IV', color='crimson', alpha=0.8)
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            axes[1,1].set_ylabel('Average IV')
            axes[1,1].set_xlabel('Risk-Free Rate Range')
            axes[1,1].set_title('Actual vs Predicted IV by Risk-Free Rate (Bar)')
            axes[1,1].legend()
            for bar, mean in zip(bars1, actual_means):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, float(mean) + float(actual_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='navy')
            for bar, mean in zip(bars2, predicted_means):
                axes[1,1].text(bar.get_x() + bar.get_width()/2, float(mean) + float(predicted_means.max()) * 0.01,
                               f'{float(mean):.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='crimson')
        
        # 6. Comparison: Predicted vs Actual IV by Time Buckets
        time_comparison = df.groupby('maturity_bucket', observed=True)[['actual_iv', 'predicted_iv']].agg(['mean', 'count']).dropna()
        if not time_comparison.empty:
            x = np.arange(len(time_comparison))
            width = 0.35
            
            bars1 = axes[1,2].bar(x - width/2, time_comparison[('actual_iv', 'mean')], width, 
                                 label='Actual IV', alpha=0.7, color='navy')
            bars2 = axes[1,2].bar(x + width/2, time_comparison[('predicted_iv', 'mean')], width,
                                 label='Predicted IV', alpha=0.7, color='crimson')
            
            axes[1,2].set_xticks(x)
            axes[1,2].set_xticklabels(time_comparison.index, rotation=45, ha='right')
            axes[1,2].set_ylabel('Average IV')
            axes[1,2].set_xlabel('Time to Maturity')
            axes[1,2].set_title('Actual vs Predicted IV by Maturity')
            axes[1,2].legend()
            
            # Add value labels on bars
            for bar, mean in zip(bars1, time_comparison[('actual_iv', 'mean')]):
                axes[1,2].text(bar.get_x() + bar.get_width()/2, mean + max(time_comparison[('actual_iv', 'mean')]) * 0.01,
                              f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            for bar, mean in zip(bars2, time_comparison[('predicted_iv', 'mean')]):
                axes[1,2].text(bar.get_x() + bar.get_width()/2, mean + max(time_comparison[('predicted_iv', 'mean')]) * 0.01,
                              f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        output_path = self._results_dir() / 'iv_detailed_breakdowns.png'
        self._ensure_parent(output_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        if self.verbose:
            print(f"Detailed IV breakdown plots saved to: {output_path}")

    def _get_comprehensive_stats(self, df):
        """Get all comprehensive statistics for enhanced TXT output"""
        stats = {}
        
        # Error metrics
        diff = df['predicted_iv'] - df['actual_iv']
        stats['error_metrics'] = {
            'MAE': float(np.mean(np.abs(diff))),
            'MSE': float(np.mean(diff**2)),
            'RMSE': float(np.sqrt(np.mean(diff**2))),
            'MAPE_pct': float(np.mean(np.abs(diff / df['actual_iv'])) * 100),
            'correlation': float(np.corrcoef(df['predicted_iv'], df['actual_iv'])[0,1])
        }
        
        # Descriptive stats
        stats['descriptive'] = {}
        for series_name in ['actual_iv', 'predicted_iv']:
            stats['descriptive'][series_name] = self._describe_series(series_name, df[series_name])
        
        # Historical vol comparisons
        stats['hist_vol'] = {}
        hist_cols = ['historical_volatility_input', 'hist_vol_10d_input', 'hist_vol_30d_input', 'hist_vol_90d_input']
        for hc in hist_cols:
            if hc in df.columns:
                aligned = df[['actual_iv', 'predicted_iv', hc]].dropna()
                if len(aligned) > 1:
                    stats['hist_vol'][hc] = {
                        'corr_actual_iv': float(np.corrcoef(aligned['actual_iv'], aligned[hc])[0,1]),
                        'corr_predicted_iv': float(np.corrcoef(aligned['predicted_iv'], aligned[hc])[0,1]),
                        'mae_actual_vs_hist': float(np.mean(np.abs(aligned['actual_iv'] - aligned[hc]))),
                        'rmse_actual_vs_hist': float(np.sqrt(np.mean((aligned['actual_iv'] - aligned[hc])**2)))
                    }
        
        # Bucket analysis
        stats['bucket_analysis'] = {}
        
        # Moneyness buckets
        mon_stats = df.groupby('moneyness_bucket', observed=True).agg({
            'actual_iv': ['mean', 'std', 'count'],
            'predicted_iv': ['mean', 'std', 'count'],
            'iv_error_bps': ['mean', 'std', 'count']
        }).dropna()
        if not mon_stats.empty:
            stats['bucket_analysis']['moneyness'] = {}
            for bucket in mon_stats.index:
                stats['bucket_analysis']['moneyness'][bucket] = {
                    'actual_iv_mean': float(mon_stats.loc[bucket, ('actual_iv', 'mean')]),
                    'predicted_iv_mean': float(mon_stats.loc[bucket, ('predicted_iv', 'mean')]),
                    'iv_error_bps_mean': float(mon_stats.loc[bucket, ('iv_error_bps', 'mean')]),
                    'count': int(mon_stats.loc[bucket, ('actual_iv', 'count')])
                }
        
        # Maturity buckets
        mat_stats = df.groupby('maturity_bucket', observed=True).agg({
            'actual_iv': ['mean', 'std', 'count'],
            'predicted_iv': ['mean', 'std', 'count'],
            'iv_error_bps': ['mean', 'std', 'count']
        }).dropna()
        if not mat_stats.empty:
            stats['bucket_analysis']['maturity'] = {}
            for bucket in mat_stats.index:
                stats['bucket_analysis']['maturity'][bucket] = {
                    'actual_iv_mean': float(mat_stats.loc[bucket, ('actual_iv', 'mean')]),
                    'predicted_iv_mean': float(mat_stats.loc[bucket, ('predicted_iv', 'mean')]),
                    'iv_error_bps_mean': float(mat_stats.loc[bucket, ('iv_error_bps', 'mean')]),
                    'count': int(mat_stats.loc[bucket, ('actual_iv', 'count')])
                }
        
        return stats

    def save_iv_summary_text(self, df):
        """Save comprehensive human-readable IV summary with all important statistics"""
        lines = ['=' * 60, 'COMPREHENSIVE IMPLIED VOLATILITY ANALYSIS SUMMARY', '=' * 60, '']

        stats = self._get_comprehensive_stats(df)

        # Diagnostics
        lines.extend(['DIAGNOSTICS', '-' * 11])
        for k, v in df.attrs.get('diagnostics', {}).items():
            lines.append(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
        lines.append('')

        # Error metrics
        lines.extend(['ERROR METRICS (Predicted vs Actual IV)', '-' * 38])
        for metric, value in stats['error_metrics'].items():
            if 'pct' in metric:
                lines.append(f"{metric.replace('_', ' ').title()}: {value:.3f}%")
            else:
                lines.append(f"{metric}: {value:.6f}")
        lines.append('')

        # Error in basis points
        lines.extend(['IV ERROR IN BASIS POINTS', '-' * 23])
        err_bps = df['iv_error_bps']
        bps_stats = [
            ('Mean', err_bps.mean()), ('Median', err_bps.median()), ('Std Dev', err_bps.std()),
            ('Min', err_bps.min()), ('Max', err_bps.max()),
            ('P2.5', np.percentile(err_bps, 2.5)), ('P97.5', np.percentile(err_bps, 97.5)),
            ('P5', np.percentile(err_bps, 5)), ('P95', np.percentile(err_bps, 95))
        ]
        
        for name, val in bps_stats:
            lines.append(f"{name} (bps): {val:.2f}")
        lines.append('')

        # Descriptive statistics
        lines.extend(['DESCRIPTIVE STATISTICS', '-' * 20])
        for series_name in ['actual_iv', 'predicted_iv']:
            lines.append(f"\n{series_name.replace('_', ' ').upper()}:")
            stats_dict = stats['descriptive'][series_name]
            key_metrics = ['count', 'mean', 'std', 'min', 'q25', 'median', 'q75', 'max', 'p01', 'p99']
            for metric in key_metrics:
                key = f'{series_name}_{metric}'
                if key in stats_dict:
                    value = stats_dict[key]
                    lines.append(f"  {metric}: {value:.6f}" if isinstance(value, float) else f"  {metric}: {value}")
        lines.append('')

        # Bucket Analysis
        lines.extend(['ANALYSIS BY CATEGORIES', '-' * 22])

        if 'moneyness' in stats['bucket_analysis']:
            lines.append('\nMONEYNESS BUCKETS:')
            for bucket, bs in stats['bucket_analysis']['moneyness'].items():
                lines.append(f"  {bucket}:")
                lines.append(f"    Count: {bs['count']:,}")
                lines.append(f"    Actual IV: {bs['actual_iv_mean']:.4f}")
                lines.append(f"    Predicted IV: {bs['predicted_iv_mean']:.4f}")
                lines.append(f"    IV Error (bps): {bs['iv_error_bps_mean']:.2f}")

        if 'maturity' in stats['bucket_analysis']:
            lines.append('\nMATURITY BUCKETS:')
            for bucket, bs in stats['bucket_analysis']['maturity'].items():
                lines.append(f"  {bucket}:")
                lines.append(f"    Count: {bs['count']:,}")
                lines.append(f"    Actual IV: {bs['actual_iv_mean']:.4f}")
                lines.append(f"    Predicted IV: {bs['predicted_iv_mean']:.4f}")
                lines.append(f"    IV Error (bps): {bs['iv_error_bps_mean']:.2f}")
        lines.append('')

        # Historical volatility comparisons
        lines.extend(['HISTORICAL VOLATILITY COMPARISONS', '-' * 33])
        if stats['hist_vol']:
            for hc, hs in stats['hist_vol'].items():
                lines.append(f"\n{hc.replace('_input', '').replace('_', ' ').upper()}:")
                lines.append(f"  Correlation with Actual IV: {hs['corr_actual_iv']:.4f}")
                lines.append(f"  Correlation with Predicted IV: {hs['corr_predicted_iv']:.4f}")
                lines.append(f"  MAE (Actual IV vs Historical): {hs['mae_actual_vs_hist']:.4f}")
                lines.append(f"  RMSE (Actual IV vs Historical): {hs['rmse_actual_vs_hist']:.4f}")
        else:
            lines.append('No historical volatility columns found in data.')
        lines.append('')

        # Performance insights
        lines.extend(['PERFORMANCE INSIGHTS', '-' * 19])
        mae = stats['error_metrics']['MAE']
        correlation = stats['error_metrics']['correlation']
        mean_error_bps = err_bps.mean()

        lines.append("EXCELLENT: Very low Mean Absolute Error (<2 vol points)" if mae < 0.02
                     else "GOOD: Low Mean Absolute Error (<5 vol points)" if mae < 0.05
                     else "ATTENTION: High Mean Absolute Error (>5 vol points)")
        lines.append("EXCELLENT: Very high correlation with actual IV" if correlation > 0.9
                     else "GOOD: High correlation with actual IV" if correlation > 0.8
                     else "ATTENTION: Low correlation with actual IV")
        lines.append("EXCELLENT: Very low systematic bias (<10 bps)" if abs(mean_error_bps) < 10
                     else "GOOD: Low systematic bias (<50 bps)" if abs(mean_error_bps) < 50
                     else "ATTENTION: High systematic bias (>50 bps)")

        lines.extend(['', '=' * 60, 'END OF ANALYSIS', '=' * 60])

        output_path = self._results_dir() / 'iv_comprehensive_summary.txt'
        self._ensure_parent(output_path)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        if self.verbose:
            print(f"Comprehensive IV summary saved to: {output_path}")

    def run_full_analysis(self):
        """Execute complete IV analysis pipeline"""
        try:
            df = self.load_and_preprocess_data()
            if df is None:
                return None
            
            df_iv = self.calculate_iv_metrics(df)
            if df_iv is None or df_iv.empty:
                print("No valid IV data after cleaning")
                return None
            
            # Generate all outputs
            self.plot_iv_analysis(df_iv)
            self.plot_volatility_smile_and_residuals(df_iv)
            self.plot_detailed_iv_breakdowns(df_iv)
            self.save_iv_summary_text(df_iv)
            
            print("IV analysis completed successfully")
            return df_iv
            
        except Exception as e:
            print(f"Analysis failed: {e}")
            return None

# Usage
if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if arg.lower() == 'all':
        run_names = IVAnalyzer.discover_run_names()
        if not run_names:
            print("No run_MLP* folders found under results/. Nothing to do.")
            sys.exit(0)
        print(f"Discovered runs: {', '.join(run_names)}")
        for rn in run_names:
            print(f"\n=== Processing {rn} ===")
            analyzer = IVAnalyzer(run_name=rn)
            analyzer.run_full_analysis()
        print("\nAll runs completed. Outputs saved under vol_results/<run_name>/")
    else:
        analyzer = IVAnalyzer(run_name=arg)
        print(f"Using run_name: {analyzer.run_name}")
        analyzer.run_full_analysis()
