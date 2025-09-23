"""
Multi-Layer Perceptron for Option Pricing
==========================================

Advanced neural network implementation for European option pricing using S&P 500 index options data.
Features normalized target training, vega-weighted loss functions, and comprehensive evaluation metrics
for enhanced pricing accuracy compared to traditional Black-Scholes models.

This implementation supports:
- Multiple network architectures with configurable layers
- Vega-weighted loss for sensitivity-based training
- Normalized target training (price/underlying)
- Advanced data filtering and quality controls
- Comprehensive performance evaluation and visualization

Author: Federico Galli
Institution: Bocconi University
Thesis: Enhancing Black–Scholes Option Pricing Accuracy with Neural Networks
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from scipy import stats
from scipy.stats import norm
from torch.optim.lr_scheduler import ReduceLROnPlateau

SEED = 42
EPS = 1e-8
DAYS_PER_YEAR = 365.25

ARCHITECTURES = {
    'small': [64, 32],
    'medium': [128, 64, 32],
    'large': [128, 128, 128],
    'extra': [256, 128, 64],
    'custom': [128, 64, 32]
}

ARCHITECTURE_NAME = 'large'
LAYER_SIZES = ARCHITECTURES[ARCHITECTURE_NAME]

LEAKY_RELU_SLOPE = 0.01
DROPOUT_RATE = 0.0
EPOCHS = 100
BATCH_SIZE = 512
LEARNING_RATE = 4e-4                # currently running on 4
WEIGHT_DECAY = 2e-4                 #
SAMPLE_SIZE = 15000000
USE_LR_WARMUP = True
WARMUP_EPOCHS = 10                  #
USE_WEIGHT_ANNEALING = True
WEIGHT_ANNEALING_EPOCHS = 10        #
EARLY_STOPPING_PATIENCE = 10                
EARLY_STOPPING_MIN_DELTA = 1e-7             
SCHEDULER_PATIENCE = 5              #        
SCHEDULER_FACTOR = 0.5                      
SCHEDULER_MIN_LR = 1e-6                     
MODEL_SAVE_PATH = 'best_model.pth'
SAVE_OPTIMIZER_STATE = True
RESULTS_FOLDER = 'results'
CREATE_TIMESTAMP_FOLDER = True
GENERATE_ALL_PLOTS = True
COMPUTE_BOOTSTRAP_CI = False
BOOTSTRAP_N_SAMPLES = 100
BOOTSTRAP_CONFIDENCE = 0.95
USE_NORMALIZED_TARGET = True
USE_ZSCORE_SCALING = False
USE_VEGA_WEIGHTED_LOSS = True                    #
USE_MULTI_DIM_STRATIFICATION = False             #
USE_MONEYNESS_SQRT_T_WEIGHTING = True            #
MONEYNESS_BINS = 12                              #
SQRT_T_BINS = 12                                 #
MIN_SQRT_T = 0.0
MAX_SQRT_T = np.sqrt(750/DAYS_PER_YEAR)
MIN_MONEYNESS = 0.5
MAX_MONEYNESS = 1.5

HUBER_DELTA = 0.1 if USE_NORMALIZED_TARGET else 1.0
USE_KAIMING_INIT = False
USE_ADAMW = True
MAX_SPREAD_RATIO = 100
SPREAD_FILTER_ENABLED = False
MIN_SPREAD_PCT = None
MAX_SPREAD_PCT = None
MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
MONEYNESS_BIN_LABELS = ['OTM\n(<0.9)', 'ATM\n(0.9-1.1)', 'ITM\n(>1.1)']
TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]  # 1M, 3M, 6M, 9M, 12M, >12M
TIME_BIN_LABELS = ['≤1M\n(≤30d)', '1-3M\n(31-91d)', '3-6M\n(92-182d)', '6-9M\n(183-274d)', '9-12M\n(275-365d)', '>12M\n(>365d)']

# Model architecture options
USE_BATCH_NORM = False
OUTPUT_ACTIVATION = 'linear'                # Options: 'linear', 'softplus', 'relu'
ENFORCE_NONNEGATIVE_AT_EVAL = False
EVAL_CLIP_MIN = 0.0

print(f"[CONFIG] Architecture: {ARCHITECTURE_NAME} {LAYER_SIZES} | BatchNorm: {USE_BATCH_NORM} | "
      f"Output: {OUTPUT_ACTIVATION} | Normalized Target: {USE_NORMALIZED_TARGET} | "
      f"Vega Weighted: {USE_VEGA_WEIGHTED_LOSS} | Multi-Dim Stratification: {USE_MULTI_DIM_STRATIFICATION}")

# Data splitting configuration
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
VAL_TEST_SPLIT = VALIDATION_RATIO / (1 - TRAIN_RATIO)
USE_TIME_BASED_SPLIT = True
USE_TIME_PERCENT_SPLIT = True
TIME_SPLIT_FRACTION = 0.7
TIME_SPLIT_DATE = '2018-01-01'          # only used if USE_TIME_PERCENT_SPLIT is false.

# Data quality and filtering parameters
ZERO_VOLUME_INCLUSION = 0.5
FILTER_ITM_OPTIONS = True
FILTER_SHORT_TERM = True
MIN_DAYS_TO_MATURITY = 0
MAX_DAYS_TO_MATURITY = 750
FILTER_VALID_SPREAD = False

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.switch_backend('Agg')
DATA_QUALITY_ISSUES = []
FEATURE_MODE = 'core_only'               # Options: 'core_only', 'selected_features', 'all_features', & for MLP2 'simplified_only', 'simplified_plus_selected', 'simplified_plus_all'


SELECTED_FEATURES = [
    "volume",
    "open_interest",
    "epu_index",
    "equity_uncertainty", 
    "equity_volatility"
    ]

# Core features (Black-Scholes features only)
CORE_FEATURES = [
    "strike_price",
    "dividend_rate",
    "risk_free_rate", 
    "days_to_maturity",
    "spx_close",
    "historical_volatility"
    ]

# Additional features (all possible extra features)
ADDITIONAL_FEATURES = [
    "volume",
    "open_interest",
    "epu_index",        
    "equity_uncertainty", #
    "equity_volatility", #
    "hist_vol_90d", #
    "hist_vol_10d", #
    "hist_vol_30d" #
    ]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed):
    """Configure random seeds for reproducible results across all libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    """Implements early stopping to prevent overfitting during neural network training.
    
    Monitors validation loss and stops training when no improvement is observed
    for a specified number of epochs (patience).
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        """Store model state when validation loss improves."""
        self.best_weights = model.state_dict().copy()

class ModelCheckpoint:
    """Handles model checkpointing during training to save the best performing model.
    
    Monitors a specified metric and saves the model when performance improves.
    """
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min', save_optimizer=True):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.best_score = None
        self.is_better = self._get_is_better_func()
        
    def _get_is_better_func(self):
        if self.mode == 'min':
            return lambda current, best: current < best
        else:
            return lambda current, best: current > best
    
    def __call__(self, current_score, model, optimizer=None, epoch=None, **kwargs):
        if self.best_score is None or self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.save_checkpoint(model, optimizer, epoch, current_score, **kwargs)
            return True
        return False
    
    def save_checkpoint(self, model, optimizer=None, epoch=None, score=None, **kwargs):
        """Save model checkpoint"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'score': score,
            **kwargs
        }
        
        if self.save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, self.filepath)
        print(f"Model checkpoint saved to {self.filepath} (score: {score:.6f})")
    
    @staticmethod
    def load_checkpoint(filepath, model, optimizer=None):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint.get('epoch', 0), checkpoint.get('score', None)

# Results Folder Management
def create_results_folder():
    """Create results folder with optional timestamp"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if CREATE_TIMESTAMP_FOLDER:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(script_dir, RESULTS_FOLDER, f"run_{timestamp}")
    else:
        results_dir = os.path.join(script_dir, RESULTS_FOLDER)
    
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def _collect_run_config() -> dict:
    """Gather key configuration for run summary."""
    cfg = dict(
        SEED=SEED,
        ARCHITECTURE_NAME=ARCHITECTURE_NAME,
        LAYER_SIZES=LAYER_SIZES,
        LEAKY_RELU_SLOPE=LEAKY_RELU_SLOPE,
        DROPOUT_RATE=DROPOUT_RATE,
        USE_BATCH_NORM=USE_BATCH_NORM,
        OUTPUT_ACTIVATION=OUTPUT_ACTIVATION,
        EPOCHS=EPOCHS,
        BATCH_SIZE=BATCH_SIZE,
        LEARNING_RATE=LEARNING_RATE,
        WEIGHT_DECAY=WEIGHT_DECAY,
        USE_LR_WARMUP=USE_LR_WARMUP,
        WARMUP_EPOCHS=WARMUP_EPOCHS,
        EARLY_STOPPING_PATIENCE=EARLY_STOPPING_PATIENCE,
        EARLY_STOPPING_MIN_DELTA=EARLY_STOPPING_MIN_DELTA,
        SCHEDULER_PATIENCE=SCHEDULER_PATIENCE,
        SCHEDULER_FACTOR=SCHEDULER_FACTOR,
        SCHEDULER_MIN_LR=SCHEDULER_MIN_LR,
        USE_NORMALIZED_TARGET=USE_NORMALIZED_TARGET,
        USE_VEGA_WEIGHTED_LOSS=USE_VEGA_WEIGHTED_LOSS,
        HUBER_DELTA=HUBER_DELTA,
        USE_ZSCORE_SCALING=USE_ZSCORE_SCALING,
        USE_ADAMW=USE_ADAMW,
        USE_MONEYNESS_SQRT_T_WEIGHTING=USE_MONEYNESS_SQRT_T_WEIGHTING,
        MONEYNESS_BINS=MONEYNESS_BINS,
        SQRT_T_BINS=SQRT_T_BINS,
        MIN_SQRT_T=MIN_SQRT_T,
        MAX_SQRT_T=MAX_SQRT_T,
        MIN_MONEYNESS=MIN_MONEYNESS,
        MAX_MONEYNESS=MAX_MONEYNESS,
        FEATURE_MODE=FEATURE_MODE,
        USE_MULTI_DIM_STRATIFICATION=USE_MULTI_DIM_STRATIFICATION,
        TRAIN_RATIO=TRAIN_RATIO,
        VAL_TEST_SPLIT=VAL_TEST_SPLIT,
        TEST_RATIO=1-VALIDATION_RATIO-TRAIN_RATIO,
        USE_TIME_BASED_SPLIT=USE_TIME_BASED_SPLIT,
        TIME_SPLIT_DATE=TIME_SPLIT_DATE,
        ZERO_VOLUME_INCLUSION=ZERO_VOLUME_INCLUSION,
        FILTER_ITM_OPTIONS=FILTER_ITM_OPTIONS,
        FILTER_SHORT_TERM=FILTER_SHORT_TERM,
        MIN_DAYS_TO_MATURITY=MIN_DAYS_TO_MATURITY,
        MAX_DAYS_TO_MATURITY=MAX_DAYS_TO_MATURITY,
        SPREAD_FILTER_ENABLED=SPREAD_FILTER_ENABLED,
        MIN_SPREAD_PCT=MIN_SPREAD_PCT,
        MAX_SPREAD_PCT=MAX_SPREAD_PCT,
    )
    return cfg

def write_run_summary(results_dir: str, extra_metrics: dict | None = None):
    """Write a summary.txt with configuration and (optionally) final metrics."""
    cfg = _collect_run_config()
    path = os.path.join(results_dir, "summary.txt")
    with open(path, "w") as f:
        f.write("=== Run Configuration Summary ===\n")
        for k, v in cfg.items():
            f.write(f"{k}: {v}\n")
        if extra_metrics:
            f.write("\n=== Final Metrics ===\n")
            for k, v in extra_metrics.items():
                f.write(f"{k}: {v}\n")
    print(f"Run summary saved to {path}")

def write_data_quality_report(results_dir: str):
    """Write data_quality_issues.txt with all collected issues (or 'None')."""
    path = os.path.join(results_dir, "data_quality_issues.txt")
    with open(path, "w") as f:
        if DATA_QUALITY_ISSUES:
            f.write("=== Data Quality Issues ===\n")
            for msg in DATA_QUALITY_ISSUES:
                f.write(f"- {msg}\n")
        else:
            f.write("No data quality issues detected.\n")
    print(f"Data quality report saved to {path}")

# Helper Functions

def setup_plot_style():
    """Set up consistent plot styling to avoid repetition."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.edgecolor': 'black',
        'axes.linewidth': 0.5
    })

def _clip_eval_predictions(y_pred: np.ndarray):
    if 'ENFORCE_NONNEGATIVE_AT_EVAL' in globals() and ENFORCE_NONNEGATIVE_AT_EVAL:
        return np.maximum(y_pred, EVAL_CLIP_MIN)
    return y_pred

# Plotting Functions

def plot_loss_curves(train_losses, val_losses, results_dir):
    """Plot and save training and validation loss curves"""
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale plot
    plt.subplot(2, 2, 2)
    plt.semilogy(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.semilogy(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.title('Loss Curves (Log Scale)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss difference
    plt.subplot(2, 2, 3)
    loss_diff = np.array(val_losses) - np.array(train_losses)
    plt.plot(loss_diff, color='purple', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation - Train Loss')
    plt.title('Overfitting Monitor', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Moving average
    plt.subplot(2, 2, 4)
    window = min(10, len(train_losses) // 5)
    if window > 1:
        train_ma = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_ma = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(train_losses)), train_ma, label=f'Train MA({window})', color='lightblue')
        plt.plot(range(window-1, len(val_losses)), val_ma, label=f'Val MA({window})', color='lightcoral')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Moving Average)')
        plt.title('Smoothed Loss Curves', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'loss_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {save_path}")

def plot_true_vs_predicted(y_true, y_pred, results_dir, df_test=None):
    """Plot comprehensive true vs predicted analysis"""
    y_pred = _clip_eval_predictions(y_pred)

    plt.figure(figsize=(12, 8))
    # True vs Predicted scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.3, s=1)
    min_val, max_val = 0, max(y_true.max(), y_pred.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel("True Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.title("True vs Predicted Prices", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Auto-fit axes to data range
    plt.xlim(0, y_true.max() * 1.1)
    plt.ylim(0, y_pred.max() * 1.1)
    
    # Residuals plot with days to maturity color coding
    plt.subplot(2, 2, 2)
    residuals = y_pred - y_true
    if df_test is not None and 'days_to_maturity' in df_test.columns:
        days_to_maturity = df_test['days_to_maturity'].values
        scatter = plt.scatter(y_true, residuals, c=days_to_maturity, alpha=0.5, s=1, cmap='viridis')
        plt.colorbar(scatter, label='Days to Maturity')
    else:
        plt.scatter(y_true, residuals, alpha=0.3, s=1)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("True Price ($)", fontsize=12)
    plt.ylabel("Residuals ($)", fontsize=12)
    plt.title("Residuals vs True Price", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Auto-fit x-axis to data range
    plt.xlim(0, y_true.max() * 1.1)
    
    # Q-Q plot for residuals normality
    plt.subplot(2, 2, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot (Residuals Normality)", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residuals vs True Price with strike price color coding
    plt.subplot(2, 2, 4)
    if df_test is not None and 'strike_price' in df_test.columns:
        strike_price = df_test['strike_price'].values
        scatter2 = plt.scatter(y_true, residuals, c=strike_price, alpha=0.5, s=1, cmap='plasma')
        plt.colorbar(scatter2, label='Strike Price ($)')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel("True Price ($)", fontsize=12)
        plt.ylabel("Residuals ($)", fontsize=12)
        plt.title("Residuals vs True Price - Colored by Strike Price", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, y_true.max() * 1.1)
    else:
        # Fallback to original MAE by price range if no strike price data
        price_bins = np.percentile(y_true, [0, 25, 50, 75, 100])
        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        
        error_by_quantile = []
        quantile_labels = []
        
        for i in range(len(price_bins)-1):
            mask = (y_true >= price_bins[i]) & (y_true < price_bins[i+1])
            if i == len(price_bins)-2:  # Include the last value
                mask = (y_true >= price_bins[i]) & (y_true <= price_bins[i+1])
            error_by_quantile.append(np.mean(np.abs(residuals[mask])))
            quantile_labels.append(f'{price_bins[i]:.2f}-{price_bins[i+1]:.2f}')
        
        plt.bar(quantile_labels, error_by_quantile, color='lightblue', alpha=0.7)
        plt.xlabel("Price Quantiles")
        plt.ylabel("Mean Absolute Error")
        plt.title("MAE by Price Range", fontweight='bold')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'prediction_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction analysis saved to {save_path}")

def plot_residuals_histograms(y_true, y_pred, results_dir, use_density=False):
    """Create histograms for residuals analysis - consolidated function with density option"""
    y_pred = _clip_eval_predictions(y_pred)
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    den = np.maximum(np.abs(y_true), EPS)
    pct_error = 100 * residuals / den
    abs_pct_error = np.abs(pct_error)
    
    plt.figure(figsize=(15, 10))
    
    # Set y-axis label based on density option
    y_label = 'Density' if use_density else 'Frequency'
    
    # 1. Absolute Residuals Distribution
    plt.subplot(2, 2, 1)
    plt.hist(abs_residuals, bins=50, alpha=0.7, color='lightcoral', edgecolor='black', density=use_density)
    plt.xlabel('Absolute Residuals ($)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title('Absolute Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals Distribution
    plt.subplot(2, 2, 2)
    plt.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=use_density)
    plt.xlabel('Residuals ($)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Percentage Residuals Distribution
    plt.subplot(2, 2, 3)
    pct_error_filtered = pct_error[(pct_error >= -200) & (pct_error <= 200)]
    plt.hist(pct_error_filtered, bins=50, alpha=0.7, color='orange', edgecolor='black', density=use_density)
    plt.xlabel('Percentage Error (%)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title('Percentage Residuals Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Absolute Percentage Error Distribution
    plt.subplot(2, 2, 4)
    abs_pct_error_filtered = abs_pct_error[abs_pct_error <= 200]
    plt.hist(abs_pct_error_filtered, bins=50, alpha=0.7, color='purple', edgecolor='black', density=use_density)
    plt.xlabel('Absolute Percentage Error (%)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title('Absolute Percentage Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0)
    suffix = '_density' if use_density else ''
    save_path = os.path.join(results_dir, f'residuals_histograms{suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals histograms{' (density)' if use_density else ''} saved to {save_path}")

def plot_residuals_vs_features(y_true, y_pred, df_test_or_X_original, results_dir):
    """Create residuals vs features scatter plots - Second PNG"""
    y_pred = _clip_eval_predictions(y_pred)

    residuals = y_pred - y_true
    
    # Handle both dataframe and matrix inputs
    if isinstance(df_test_or_X_original, pd.DataFrame):
        df_test = df_test_or_X_original
        spx_close = df_test['spx_close'].values if 'spx_close' in df_test.columns else np.full(len(df_test), np.nan)
        strike_price = df_test['strike_price'].values if 'strike_price' in df_test.columns else np.full(len(df_test), np.nan)
        days_to_maturity = df_test['days_to_maturity'].values if 'days_to_maturity' in df_test.columns else np.full(len(df_test), np.nan)
        historical_volatility = df_test['historical_volatility'].values if 'historical_volatility' in df_test.columns else np.full(len(df_test), np.nan)
    else:
        # Legacy matrix input - use centralized feature extraction
        features = extract_features_from_test_data(df_test_or_X_original)
        spx_close = features['spx_close']
        strike_price = features['strike_price']
        days_to_maturity = features['days_to_maturity']
        historical_volatility = features['historical_volatility']
    
    plt.figure(figsize=(16, 12))
    
    # 1. Residuals vs Fitted Values (Log Scale)
    plt.subplot(2, 2, 1)
    # Filter out zero and negative values for log scale
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
    
    # Helper function to create binned bar chart in current subplot
    def create_subplot_bar_chart(data, residuals, title, xlabel, colormap='viridis', num_bins=20):
        # Create bins for the data
        bin_edges = np.linspace(data.min(), data.max(), num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Assign data points to bins
        bin_indices = np.digitize(data, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        # Calculate average residuals and counts for each bin
        avg_residuals = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)
        
        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                avg_residuals[i] = np.mean(np.abs(residuals[mask]))
                bin_counts[i] = np.sum(mask)
        
        # Create color map based on bin counts
        norm = plt.Normalize(vmin=bin_counts.min(), vmax=bin_counts.max())
        cmap = plt.get_cmap(colormap)
        colors = cmap(norm(bin_counts))
        
        # Create bar chart
        bars = plt.bar(bin_centers, avg_residuals, width=(bin_edges[1] - bin_edges[0]) * 0.8, 
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('Average Absolute Residuals ($)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Number of Data Points', fontsize=10)
    
    # 2. Percentage Error vs Fitted Values (not log scale)
    plt.subplot(2, 2, 2)
    percentage_error = (residuals / np.maximum(y_true, 1.0)) * 100
    plt.scatter(y_pred, percentage_error, alpha=0.3, s=1, color='blue')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Fitted Values ($)', fontsize=12)
    plt.ylabel('Percentage Error (%)', fontsize=12)
    plt.title('Percentage Error vs Fitted Values', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. Average Absolute Residuals vs Days to Maturity (with color bar)
    plt.subplot(2, 2, 3)
    create_subplot_bar_chart(days_to_maturity, residuals, 'Average Absolute Residuals vs Days to Maturity', 'Days to Maturity', 'plasma')
    
    # 4. Average Absolute Residuals vs Historical Volatility (binned bar chart)
    plt.subplot(2, 2, 4)
    create_subplot_bar_chart(historical_volatility, residuals, 'Average Absolute Residuals vs Historical Volatility', 'Historical Volatility', 'cividis')
    
    plt.tight_layout(pad=3.0)
    save_path = os.path.join(results_dir, 'residuals_vs_features.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Residuals vs features plots saved to {save_path}")


def plot_model_performance_summary(train_losses, val_losses, y_true, y_pred, results_dir, metrics):
    """Create a comprehensive model performance summary"""
    plt.figure(figsize=(16, 12))
    y_pred = _clip_eval_predictions(y_pred)
    # Loss curves
    plt.subplot(3, 3, 1)
    plt.plot(train_losses, label='Train', color='blue')
    plt.plot(val_losses, label='Validation', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final metrics text
    plt.subplot(3, 3, 2)
    plt.axis('off')
    metrics_text = f"""Final Metrics:
    MSE: {metrics['mse']:.4f}
    MAE: {metrics['mae']:.4f}
    R²: {metrics['r2']:.4f}
    
    Best Val Loss: {min(val_losses):.6f}
    Final Epoch: {len(train_losses)}
    
    Architecture: {LAYER_SIZES}
    Learning Rate: {LEARNING_RATE}
    Batch Size: {BATCH_SIZE}"""
    plt.text(0.1, 0.9, metrics_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # True vs Predicted (simplified)
    plt.subplot(3, 3, 3)
    plt.scatter(y_true, y_pred, alpha=0.3, s=1)
    min_val, max_val = 0, max(y_true.max(), y_pred.max()) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel("True Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.title("Predictions vs. Reality", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    # Auto-fit axes to data range
    plt.xlim(0, y_true.max() * 1.1)
    plt.ylim(0, y_pred.max() * 1.1)
    
    # Error distribution
    plt.subplot(3, 3, 4)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, alpha=0.7, color='skyblue')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution", fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Additional metrics
    plt.subplot(3, 3, 5)
    plt.axis('off')
    additional_text = f"""Additional Metrics:
    
Max Absolute Error: ${np.max(np.abs(errors)):.2f}
Min Absolute Error: ${np.min(np.abs(errors)):.2f}
Error Std Dev: ${np.std(errors):.2f}
    
Price Range: ${y_true.min():.2f} - ${y_true.max():.2f}"""
    
    plt.text(0.1, 0.9, additional_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Model architecture visualization
    plt.subplot(3, 3, 6)
    input_layer_size = len(ALL_FEATURES)
    layer_sizes = [input_layer_size] + LAYER_SIZES + [1]  # Input + hidden + output
    x_pos = np.arange(len(layer_sizes))
    plt.bar(x_pos, layer_sizes, color='lightcoral', alpha=0.7)
    plt.xlabel('Layer')
    plt.ylabel('Neurons')
    plt.title('Network Architecture', fontweight='bold')
    plt.xticks(x_pos, [f'Input'] + [f'H{i+1}' for i in range(len(LAYER_SIZES))] + ['Output'])
    plt.grid(True, alpha=0.3)
    
    # Performance by price range
    plt.subplot(3, 3, 7)
    price_ranges = np.percentile(y_true, [0, 20, 40, 60, 80, 100])
    range_labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    mae_by_range = []
    
    for i in range(len(price_ranges)-1):
        mask = (y_true >= price_ranges[i]) & (y_true < price_ranges[i+1])
        if i == len(price_ranges)-2:
            mask = (y_true >= price_ranges[i]) & (y_true <= price_ranges[i+1])
        mae_by_range.append(np.mean(np.abs(errors[mask])))
    
    plt.bar(range_labels, mae_by_range, color='lightgreen', alpha=0.7)
    plt.xlabel('Price Percentile Range')
    plt.ylabel('Mean Absolute Error')
    plt.title('Performance by Price Range', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Training efficiency
    plt.subplot(3, 3, 8)
    improvement_rate = []
    for i in range(1, len(val_losses)):
        if val_losses[i-1] != 0:
            improvement = (val_losses[i-1] - val_losses[i]) / val_losses[i-1] * 100
            improvement_rate.append(improvement)
        else:
            improvement_rate.append(0)
    
    plt.plot(improvement_rate, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Improvement (%)')
    plt.title('Learning Efficiency', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Model complexity vs performance
    plt.subplot(3, 3, 9)
    # Calculate total parameters properly: weights + biases for all layers
    # Input layer: input_size * first_hidden + first_hidden (bias)
    # Hidden layers: prev_size * curr_size + curr_size (bias) 
    # Output layer: last_hidden * 1 + 1 (bias)
    input_size = _get_input_size_for_feature_mode()
    if len(LAYER_SIZES) > 0:
        total_params = input_size * LAYER_SIZES[0] + LAYER_SIZES[0]  # Input to first hidden
        for i in range(len(LAYER_SIZES)-1):
            total_params += LAYER_SIZES[i] * LAYER_SIZES[i+1] + LAYER_SIZES[i+1]  # Hidden to hidden
        total_params += LAYER_SIZES[-1] * 1 + 1  # Last hidden to output
    else:
        total_params = input_size * 1 + 1  # Direct input to output
    complexity_metrics = {
        'Parameters': total_params,
        'Layers': len(LAYER_SIZES),
        'Max Layer Size': max(LAYER_SIZES),
        'R² Score': metrics['r2']
    }
    
    plt.axis('off')
    complexity_text = "Model Complexity:\n\n"
    for key, value in complexity_metrics.items():
        if key == 'R² Score':
            complexity_text += f"{key}: {value:.4f}\n"
        else:
            complexity_text += f"{key}: {value}\n"
    
    plt.text(0.1, 0.9, complexity_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'model_performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    extra = {"MAE": f"{metrics['mae']:.6f}", "MSE": f"{metrics['mse']:.6f}", "R2": f"{metrics['r2']:.6f}"}
    write_run_summary(results_dir, extra_metrics=extra)
    write_data_quality_report(results_dir)
    print(f"Performance summary saved to {save_path}")


# Consolidated Utility Functions

def get_lr_with_warmup(current_epoch, warmup_epochs):
    """Calculate learning rate with linear warmup."""
    if current_epoch < warmup_epochs:
        return LEARNING_RATE * (current_epoch + 1) / warmup_epochs
    else:
        return LEARNING_RATE

def get_weight_annealing_factor(current_epoch, annealing_epochs):
    """Calculate weight annealing factor for gradual weight ramp-up.
    
    Args:
        current_epoch: Current training epoch (0-indexed)
        annealing_epochs: Number of epochs over which to ramp up weights
        
    Returns:
        Float between 0 and 1, where 0 means no weighting (all weights = 1)
        and 1 means full weighting
    """
    if not USE_WEIGHT_ANNEALING:
        return 1.0
    
    if current_epoch < annealing_epochs:
        # Linear ramp from 0 to 1 over annealing_epochs
        return current_epoch / annealing_epochs
    else:
        return 1.0

def apply_sample_weight_annealing(sample_weights, annealing_factor):
    """Apply annealing to sample weights by blending with uniform weights.
    
    Args:
        sample_weights: Original computed sample weights
        annealing_factor: Factor between 0 and 1 for annealing
        
    Returns:
        Annealed sample weights
    """
    if not USE_WEIGHT_ANNEALING or annealing_factor >= 1.0:
        return sample_weights
    
    # Blend between uniform weights (1.0) and computed weights
    # annealing_factor = 0 means all weights = 1, annealing_factor = 1 means full weighting
    return 1.0 + annealing_factor * (sample_weights - 1.0)

def calculate_black_scholes_vega(S, K, r, q, T, sigma):
    """Calculate Black-Scholes vega for option sensitivity analysis.
    
    Computes the sensitivity of option price to changes in implied volatility.
    Returns relative vega (vega/S) when using normalized targets, otherwise dollar vega.
    
    Args:
        S: Underlying asset price
        K: Strike price
        r: Risk-free rate
        q: Dividend yield
        T: Time to expiration (in years)
        sigma: Volatility
        
    Returns:
        Vega values with numerical stability safeguards applied
    """
    # Apply numerical floors for computational stability
    T = np.maximum(T, 1.0 / DAYS_PER_YEAR)  # Minimum 1 day
    sigma = np.maximum(sigma, 0.01)  # Minimum 1% volatility
    S = np.maximum(S, 1e-12)
    K = np.maximum(K, 1e-12)

    # Calculate d1 with clipping to prevent overflow
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d1 = np.clip(d1, -10.0, 10.0)

    nd1 = norm.pdf(d1)
    vega = S * np.exp(-q * T) * np.sqrt(T) * nd1

    if USE_NORMALIZED_TARGET:
        vega = vega / S

    return np.nan_to_num(vega, nan=0.0, posinf=0.0, neginf=0.0)


# Advanced Option Pricing Plots

def plot_normalized_error_histogram(y_true, y_pred, df_test, results_dir):
    # Plot (pred-mid)/(half_spread) histogram - more practical than raw MAE

    print("Generating normalized error histogram...")
    
    # Calculate half spread
    bid_prices = df_test['best_bid'].values
    ask_prices = df_test['best_offer'].values
    half_spread = (ask_prices - bid_prices) / 2
    
    # Calculate normalized error
    error = y_pred - y_true
    normalized_error = error / np.maximum(half_spread, 0.01)  # Floor to prevent division by zero
    
    # Filter extreme outliers for better visualization
    filtered_error = normalized_error[(normalized_error >= -10) & (normalized_error <= 10)]
    
    plt.figure(figsize=(12, 8))
    
    # Main histogram
    plt.subplot(2, 2, 1)
    plt.hist(filtered_error, bins=100, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    plt.axvline(x=np.mean(filtered_error), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(filtered_error):.3f}')
    plt.xlabel('(Predicted - Actual) / Half Spread', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Normalized Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(2, 2, 2)
    sorted_errors = np.sort(np.abs(filtered_error))
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cumulative, color='green', linewidth=2)
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='1 Half Spread')
    plt.axvline(x=2, color='orange', linestyle='--', alpha=0.7, label='2 Half Spreads')
    plt.xlabel('|Normalized Error|', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of |Normalized Error|', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics text
    plt.subplot(2, 2, 3)
    plt.axis('off')
    stats_text = f"""Normalized Error Statistics:
    
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
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    # Box plot by price ranges
    plt.subplot(2, 2, 4)
    price_quartiles = pd.qcut(y_true, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    quartile_data = []
    quartile_labels = []
    
    # Create a mask for the filtered errors (those that weren't filtered out as outliers)
    non_outlier_mask = (normalized_error >= -10) & (normalized_error <= 10)
    
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        # Get the mask for this quartile, considering only non-outlier points
        mask = (price_quartiles == q) & non_outlier_mask
        if mask.sum() > 0:
            # Get the errors for this quartile (already filtered for non-outliers)
            quartile_errors = normalized_error[mask]
            quartile_data.append(quartile_errors)
            quartile_labels.append(f'{q}\n(n={len(quartile_errors)})')
    
    plt.boxplot(quartile_data, labels=quartile_labels)
    plt.ylabel('Normalized Error', fontsize=12)
    plt.xlabel('Price Quartiles', fontsize=12)
    plt.title('Normalized Error by Price Range', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'normalized_error_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized error histogram saved to {save_path}")

def plot_vega_weighted_error(y_true, y_pred, vega_weights, results_dir):

    y_pred = _clip_eval_predictions(y_pred)
    if vega_weights is None:
        print("Vega weights not available - skipping vega-weighted error plot")
        return
        
    print("Generating vega-weighted error analysis...")
    
    abs_error = np.abs(y_pred - y_true)
    
    # Calculate vega-weighted metrics
    vmin = 1e-3 if USE_NORMALIZED_TARGET else 0.1
    vega_clipped = np.maximum(vega_weights, vmin)
    
    vega_weighted_error = abs_error * vega_weights
    vega_normalized_error = abs_error / vega_clipped
    
    plt.figure(figsize=(15, 10))
    
    # Vega vs Absolute Error scatter
    plt.subplot(2, 3, 1)
    plt.scatter(vega_weights, abs_error, alpha=0.3, s=1, c='blue')
    plt.xlabel('Vega', fontsize=12)
    plt.ylabel('Absolute Error ($)', fontsize=12)
    plt.title('Absolute Error vs Vega', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xscale('log')
    
    # Vega-weighted error distribution
    plt.subplot(2, 3, 2)
    plt.hist(vega_weighted_error, bins=50, alpha=0.7, color='green', edgecolor='black', density=True)
    plt.xlabel('|Error| × Vega', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Vega-Weighted Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Vega-normalized error distribution
    plt.subplot(2, 3, 3)
    vega_norm_filtered = vega_normalized_error[vega_normalized_error <= np.percentile(vega_normalized_error, 95)]
    plt.hist(vega_norm_filtered, bins=50, alpha=0.7, color='orange', edgecolor='black', density=True)
    plt.xlabel('|Error| / max(Vega, vmin)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Vega-Normalized Error Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Vega percentile analysis
    plt.subplot(2, 3, 4)
    vega_percentiles = [10, 25, 50, 75, 90, 95, 99]
    vega_thresholds = np.percentile(vega_weights, vega_percentiles)
    
    mean_errors = []
    vega_weighted_means = []
    vega_norm_means = []
    
    for i, threshold in enumerate(vega_thresholds):
        if i == 0:
            mask = vega_weights <= threshold
            label = f'≤P{vega_percentiles[i]}'
        else:
            mask = (vega_weights > vega_thresholds[i-1]) & (vega_weights <= threshold)
            label = f'P{vega_percentiles[i-1]}-P{vega_percentiles[i]}'
        
        if mask.sum() > 0:
            mean_errors.append(np.mean(abs_error[mask]))
            vega_weighted_means.append(np.mean(vega_weighted_error[mask]))
            vega_norm_means.append(np.mean(vega_normalized_error[mask]))
    
    x_pos = np.arange(len(mean_errors))
    width = 0.25
    
    plt.bar(x_pos - width, mean_errors, width, label='Mean |Error|', alpha=0.7, color='blue')
    plt.bar(x_pos, vega_weighted_means, width, label='Mean |Error|×Vega', alpha=0.7, color='green')
    plt.bar(x_pos + width, vega_norm_means, width, label='Mean |Error|/Vega', alpha=0.7, color='orange')
    
    plt.xlabel('Vega Percentile Ranges', fontsize=12)
    plt.ylabel('Error Metrics', fontsize=12)
    plt.title('Error Metrics by Vega Percentiles', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'P{vega_percentiles[i-1] if i > 0 else 0}-P{vega_percentiles[i]}' for i in range(len(mean_errors))], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    overall_vega_weighted = np.mean(vega_weighted_error)
    overall_vega_normalized = np.mean(vega_normalized_error)
    high_vega_mask = vega_weights >= np.percentile(vega_weights, 75)
    low_vega_mask = vega_weights <= np.percentile(vega_weights, 25)
    
    stats_text = f"""Vega-Weighted Error Analysis:
    
Overall Metrics:
Mean |Error| × Vega: {overall_vega_weighted:.6f}
Mean |Error| / Vega: {overall_vega_normalized:.4f}
    
High Vega (P75+) Options:
Count: {high_vega_mask.sum():,}
Mean |Error|: ${np.mean(abs_error[high_vega_mask]):.4f}
Mean Vega: {np.mean(vega_weights[high_vega_mask]):.4f}
    
Low Vega (P25-) Options:
Count: {low_vega_mask.sum():,}
Mean |Error|: ${np.mean(abs_error[low_vega_mask]):.4f}
Mean Vega: {np.mean(vega_weights[low_vega_mask]):.4f}
    
Vega Range: {vega_weights.min():.6f} - {vega_weights.max():.6f}"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Correlation analysis
    plt.subplot(2, 3, 6)
    
    # Create bins for vega and calculate mean error in each bin
    vega_bins = np.logspace(np.log10(vega_weights.min()), np.log10(vega_weights.max()), 20)
    bin_centers = []
    bin_mean_errors = []
    bin_counts = []
    
    for i in range(len(vega_bins)-1):
        mask = (vega_weights >= vega_bins[i]) & (vega_weights < vega_bins[i+1])
        if mask.sum() > 0:
            bin_centers.append(np.sqrt(vega_bins[i] * vega_bins[i+1]))  # Geometric mean
            bin_mean_errors.append(np.mean(abs_error[mask]))
            bin_counts.append(mask.sum())
    
    # Color by count
    scatter = plt.scatter(bin_centers, bin_mean_errors, c=bin_counts, s=50, alpha=0.7, cmap='viridis')
    plt.colorbar(scatter, label='Number of Options')
    plt.xlabel('Vega (log scale)', fontsize=12)
    plt.ylabel('Mean Absolute Error ($)', fontsize=12)
    plt.title('Mean Error vs Vega (Binned)', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'vega_weighted_error_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Vega-weighted error analysis saved to {save_path}")

def plot_error_pivot_grid(y_true, y_pred, df_test_or_X_original, results_dir):
    """Pivot mean abs % error on a grid - beats separate bar charts and shows smile/term-structure weak spots"""
    print("Generating error pivot grid...")
    
    y_pred = _clip_eval_predictions(y_pred)
    
    # Handle both dataframe and matrix inputs
    if isinstance(df_test_or_X_original, pd.DataFrame):
        df_test = df_test_or_X_original
        spx_close = df_test['spx_close'].values if 'spx_close' in df_test.columns else np.full(len(df_test), np.nan)
        strike_price = df_test['strike_price'].values if 'strike_price' in df_test.columns else np.full(len(df_test), np.nan)
        days_to_maturity = df_test['days_to_maturity'].values if 'days_to_maturity' in df_test.columns else np.full(len(df_test), np.nan)
    else:
        # Legacy matrix input - use centralized feature extraction
        features = extract_features_from_test_data(df_test_or_X_original)
        spx_close = features['spx_close']
        strike_price = features['strike_price']
        days_to_maturity = features['days_to_maturity']
    
    # Calculate metrics
    moneyness = spx_close / strike_price
    time_to_expiration = days_to_maturity / DAYS_PER_YEAR  # Convert days to years
    
    # Create finer bins for the pivot grid
    moneyness_bins = np.linspace(0.7, 1.3, 13)  
    time_bins = np.linspace(0, 2, 9)  
    
    moneyness_labels = [f'{moneyness_bins[i]:.2f}-{moneyness_bins[i+1]:.2f}' for i in range(len(moneyness_bins)-1)]
    time_labels = [f'{time_bins[i]:.2f}-{time_bins[i+1]:.2f}' for i in range(len(time_bins)-1)]
    
    # Create pivot table
    pivot_data = np.full((len(time_bins)-1, len(moneyness_bins)-1), np.nan)
    pivot_counts = np.zeros((len(time_bins)-1, len(moneyness_bins)-1))
    
    for i in range(len(time_bins)-1):
        for j in range(len(moneyness_bins)-1):
            time_mask = (time_to_expiration >= time_bins[i]) & (time_to_expiration < time_bins[i+1])
            money_mask = (moneyness >= moneyness_bins[j]) & (moneyness < moneyness_bins[j+1])
            combined_mask = time_mask & money_mask
            
            if combined_mask.sum() >= 10:  # Minimum sample size
                pivot_data[i, j] = np.mean(np.abs((y_pred[combined_mask] - y_true[combined_mask]) / np.maximum(y_true[combined_mask], 1.0)) * 100)
                pivot_counts[i, j] = combined_mask.sum()
    
    plt.figure(figsize=(16, 12))
    
    # Main heatmap
    plt.subplot(2, 2, 1)
    im1 = plt.imshow(pivot_data, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    plt.colorbar(im1, label='Mean Absolute % Error')
    plt.xlabel('Moneyness (S/K)', fontsize=12)
    plt.ylabel('Time to Expiration (Years)', fontsize=12)
    plt.title('Mean Absolute % Error Heatmap', fontsize=14, fontweight='bold')
    
    # Set custom ticks
    plt.xticks(range(len(moneyness_labels))[::2], [moneyness_labels[i] for i in range(0, len(moneyness_labels), 2)], rotation=45)
    plt.yticks(range(len(time_labels))[::1], [time_labels[i] for i in range(0, len(time_labels), 1)])
    
    # Add text annotations for values
    for i in range(len(time_bins)-1):
        for j in range(len(moneyness_bins)-1):
            if not np.isnan(pivot_data[i, j]):
                plt.text(j, i, f'{pivot_data[i, j]:.1f}', ha='center', va='center', 
                        color='white' if pivot_data[i, j] > np.nanmean(pivot_data) else 'black', fontsize=8)
    
    # Sample count heatmap
    plt.subplot(2, 2, 2)
    im2 = plt.imshow(pivot_counts, cmap='Blues', aspect='auto', interpolation='nearest')
    plt.colorbar(im2, label='Sample Count')
    plt.xlabel('Moneyness (S/K)', fontsize=12)
    plt.ylabel('Time to Expiration (Years)', fontsize=12)
    plt.title('Sample Count Heatmap', fontsize=14, fontweight='bold')
    
    plt.xticks(range(len(moneyness_labels))[::2], [moneyness_labels[i] for i in range(0, len(moneyness_labels), 2)], rotation=45)
    plt.yticks(range(len(time_labels))[::1], [time_labels[i] for i in range(0, len(time_labels), 1)])
    
    # Add count annotations
    for i in range(len(time_bins)-1):
        for j in range(len(moneyness_bins)-1):
            if pivot_counts[i, j] > 0:
                plt.text(j, i, f'{int(pivot_counts[i, j])}', ha='center', va='center', 
                        color='white' if pivot_counts[i, j] > np.mean(pivot_counts) else 'black', fontsize=8)
    
    # Moneyness profile (average across time)
    plt.subplot(2, 2, 3)
    moneyness_profile = np.nanmean(pivot_data, axis=0)
    moneyness_centers = [(moneyness_bins[i] + moneyness_bins[i+1])/2 for i in range(len(moneyness_bins)-1)]
    
    valid_mask = ~np.isnan(moneyness_profile)
    plt.plot(np.array(moneyness_centers)[valid_mask], moneyness_profile[valid_mask], 'o-', linewidth=2, markersize=6, color='blue')
    plt.xlabel('Moneyness (S/K)', fontsize=12)
    plt.ylabel('Mean Absolute % Error', fontsize=12)
    plt.title('Error Profile by Moneyness', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='ATM')
    plt.legend()
    
    # Time profile (average across moneyness)
    plt.subplot(2, 2, 4)
    time_profile = np.nanmean(pivot_data, axis=1)
    time_centers = [(time_bins[i] + time_bins[i+1])/2 for i in range(len(time_bins)-1)]
    
    valid_mask = ~np.isnan(time_profile)
    plt.plot(np.array(time_centers)[valid_mask], time_profile[valid_mask], 'o-', linewidth=2, markersize=6, color='green')
    plt.xlabel('Time to Expiration (Years)', fontsize=12)
    plt.ylabel('Mean Absolute % Error', fontsize=12)
    plt.title('Error Profile by Time to Expiration', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'error_pivot_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Error pivot grid saved to {save_path}")
    
    # Save pivot data to CSV for further analysis
    pivot_df = pd.DataFrame(pivot_data, 
                           index=[f'T_{label}' for label in time_labels],
                           columns=[f'M_{label}' for label in moneyness_labels])
    
    counts_df = pd.DataFrame(pivot_counts,
                           index=[f'T_{label}' for label in time_labels], 
                           columns=[f'M_{label}' for label in moneyness_labels])
    
    pivot_csv_path = os.path.join(results_dir, 'error_pivot_data.csv')
    counts_csv_path = os.path.join(results_dir, 'error_pivot_counts.csv')
    
    pivot_df.to_csv(pivot_csv_path)
    counts_df.to_csv(counts_csv_path)
    
    print(f"Pivot data saved to {pivot_csv_path}")
    print(f"Pivot counts saved to {counts_csv_path}")


# Bootstrap Confidence Intervals

def compute_bootstrap_confidence_intervals(y_true, y_pred, results_dir):
    """Compute bootstrap confidence intervals for multiple metrics"""
    if not COMPUTE_BOOTSTRAP_CI:
        print("Bootstrap CI computation disabled")
        return
    
    print(f"\nComputing bootstrap confidence intervals...")
    print(f"Bootstrap samples: {BOOTSTRAP_N_SAMPLES}")
    print(f"Confidence level: {BOOTSTRAP_CONFIDENCE*100:.0f}%")
    
    n_samples = len(y_true)
    alpha = 1 - BOOTSTRAP_CONFIDENCE
    lower_percentile = (alpha/2) * 100
    upper_percentile = (1 - alpha/2) * 100
    
    # Initialize arrays to store bootstrap statistics
    bootstrap_mae = []
    bootstrap_mse = []
    bootstrap_rmse = []
    bootstrap_r2 = []
    bootstrap_mape = []
    
    # Perform bootstrap sampling
    np.random.seed(SEED)
    for i in range(BOOTSTRAP_N_SAMPLES):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute metrics for this bootstrap sample
        mae_boot = mean_absolute_error(y_true_boot, y_pred_boot)
        mse_boot = mean_squared_error(y_true_boot, y_pred_boot)
        rmse_boot = np.sqrt(mse_boot)
        r2_boot = r2_score(y_true_boot, y_pred_boot)
        den = np.maximum(np.abs(y_true_boot), EPS)
        mape_boot = np.mean(np.abs((y_true_boot - y_pred_boot) / den)) * 100
        
        bootstrap_mae.append(mae_boot)
        bootstrap_mse.append(mse_boot)
        bootstrap_rmse.append(rmse_boot)
        bootstrap_r2.append(r2_boot)
        bootstrap_mape.append(mape_boot)
    
    # Convert to numpy arrays
    bootstrap_mae = np.array(bootstrap_mae)
    bootstrap_mse = np.array(bootstrap_mse)
    bootstrap_rmse = np.array(bootstrap_rmse)
    bootstrap_r2 = np.array(bootstrap_r2)
    bootstrap_mape = np.array(bootstrap_mape)
    
    # Compute confidence intervals
    metrics_ci = {
        'MAE': {
            'lower': np.percentile(bootstrap_mae, lower_percentile),
            'upper': np.percentile(bootstrap_mae, upper_percentile),
            'mean': np.mean(bootstrap_mae),
            'std': np.std(bootstrap_mae)
        },
        'MSE': {
            'lower': np.percentile(bootstrap_mse, lower_percentile),
            'upper': np.percentile(bootstrap_mse, upper_percentile),
            'mean': np.mean(bootstrap_mse),
            'std': np.std(bootstrap_mse)
        },
        'RMSE': {
            'lower': np.percentile(bootstrap_rmse, lower_percentile),
            'upper': np.percentile(bootstrap_rmse, upper_percentile),
            'mean': np.mean(bootstrap_rmse),
            'std': np.std(bootstrap_rmse)
        },
        'R²': {
            'lower': np.percentile(bootstrap_r2, lower_percentile),
            'upper': np.percentile(bootstrap_r2, upper_percentile),
            'mean': np.mean(bootstrap_r2),
            'std': np.std(bootstrap_r2)
        },
        'MAPE': {
            'lower': np.percentile(bootstrap_mape, lower_percentile),
            'upper': np.percentile(bootstrap_mape, upper_percentile),
            'mean': np.mean(bootstrap_mape),
            'std': np.std(bootstrap_mape)
        }
    }
    
    # Print results
    print(f"\n--- Bootstrap Confidence Intervals ({BOOTSTRAP_CONFIDENCE*100:.0f}%) ---")
    for metric, ci in metrics_ci.items():
        print(f"{metric:>6}: [{ci['lower']:8.4f}, {ci['upper']:8.4f}] (mean: {ci['mean']:8.4f}, std: {ci['std']:8.4f})")
    
    # Save results to CSV
    ci_df = pd.DataFrame(metrics_ci).T
    ci_df.index.name = 'Metric'
    ci_path = os.path.join(results_dir, 'bootstrap_confidence_intervals.csv')
    ci_df.to_csv(ci_path)
    print(f"Bootstrap CI results saved to {ci_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    bootstrap_data = [bootstrap_mae, bootstrap_mse, bootstrap_rmse, bootstrap_r2, bootstrap_mape]
    metric_names = ['MAE', 'MSE', 'RMSE', 'R²', 'MAPE']
    
    for i, (data, name) in enumerate(zip(bootstrap_data, metric_names)):
        if i < len(axes):
            axes[i].hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].axvline(metrics_ci[name]['lower'], color='red', linestyle='--', 
                           label=f'{lower_percentile:.1f}th percentile')
            axes[i].axvline(metrics_ci[name]['upper'], color='red', linestyle='--', 
                           label=f'{upper_percentile:.1f}th percentile')
            axes[i].axvline(metrics_ci[name]['mean'], color='green', linestyle='-', 
                           label='Mean')
            axes[i].set_title(f'{name} Bootstrap Distribution')
            axes[i].set_xlabel(name)
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(bootstrap_data) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    bootstrap_plot_path = os.path.join(results_dir, 'bootstrap_distributions.png')
    plt.savefig(bootstrap_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Bootstrap distributions plot saved to {bootstrap_plot_path}")
    
    return metrics_ci


# Neural Network Model

class OptionPricingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, vega_weights: np.ndarray = None, sample_weights: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # Add vega weights if provided (for vega-weighted loss)
        if vega_weights is not None:
            self.vega_weights = torch.tensor(vega_weights, dtype=torch.float32).view(-1, 1)
        else:
            self.vega_weights = None
            
        # Add sample weights if provided (for moneyness×√T weighting)
        if sample_weights is not None:
            self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32).view(-1, 1)
        else:
            self.sample_weights = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Always return a 4-tuple; make sure default weights have shape [1] so dataloader stacks to [batch,1]
        if self.vega_weights is not None:
            vega_w = self.vega_weights[idx]
        else:
            vega_w = torch.ones(1, dtype=torch.float32)  # shape [1], not scalar

        if self.sample_weights is not None:
            sample_w = self.sample_weights[idx]
        else:
            sample_w = torch.ones(1, dtype=torch.float32)  # shape [1], not scalar

        return self.X[idx], self.y[idx], vega_w, sample_w


# Moneyness × √T Sample Weighting

def compute_moneyness_sqrt_t_weights(df):
    """Compute sample weights based on moneyness and time-to-expiration distribution.
    
    Creates balanced training by weighting samples inversely proportional to their
    density in the moneyness × √T space, helping the model learn from all regions
    of the option space more effectively.
    
    Args:
        df: DataFrame containing option data with required columns
        
    Returns:
        Array of normalized sample weights (mean=1) or None if disabled
    """
    if not USE_MONEYNESS_SQRT_T_WEIGHTING:
        return None
    
    # Check if required columns exist
    required_cols = ['spx_close', 'strike_price', 'days_to_maturity']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Moneyness × √T weighting disabled.")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Engineer features
    moneyness = df['spx_close'].values / df['strike_price'].values  # S/K moneyness
    sqrt_t = np.sqrt(df['days_to_maturity'].values / DAYS_PER_YEAR)  # Convert days to years, then sqrt
    
    print(f"Moneyness range: {moneyness.min():.4f} - {moneyness.max():.4f}")
    print(f"√T range: {sqrt_t.min():.4f} - {sqrt_t.max():.4f}")
    
    # Clip to reasonable bounds
    moneyness = np.clip(moneyness, MIN_MONEYNESS, MAX_MONEYNESS)
    sqrt_t = np.clip(sqrt_t, MIN_SQRT_T, MAX_SQRT_T)
    
    # Create 2D bins
    moneyness_edges = np.linspace(MIN_MONEYNESS, MAX_MONEYNESS, MONEYNESS_BINS + 1)
    sqrt_t_edges = np.linspace(MIN_SQRT_T, MAX_SQRT_T, SQRT_T_BINS + 1)
    
    # Assign each sample to a bin
    moneyness_bin_indices = np.digitize(moneyness, moneyness_edges) - 1
    sqrt_t_bin_indices = np.digitize(sqrt_t, sqrt_t_edges) - 1
    
    # Clip to valid bin ranges
    moneyness_bin_indices = np.clip(moneyness_bin_indices, 0, MONEYNESS_BINS - 1)
    sqrt_t_bin_indices = np.clip(sqrt_t_bin_indices, 0, SQRT_T_BINS - 1)

    
    # Vectorized 2D histogram computation for better performance
    bin_counts, _, _ = np.histogram2d(moneyness, sqrt_t, 
                                     bins=[moneyness_edges, sqrt_t_edges])

    sample_weights = np.zeros(len(moneyness))
    valid_bins = bin_counts > 0

    if not np.any(valid_bins):
        print("Warning: All 2D bins are empty after clipping—check MIN/MAX and bins settings.")
        return np.ones(len(moneyness))  # Return uniform weights as fallback

    mean_count = bin_counts[valid_bins].mean()

    # Create weight lookup table based on relative inverse frequency with power 0.5
    weight_lookup = np.zeros_like(bin_counts, dtype=float)
    weight_lookup[valid_bins] = (mean_count / bin_counts[valid_bins]) ** 0.5
    weight_lookup[~valid_bins] = 1.0  # Fallback for empty bins (neutral weight)

    # Vectorized weight assignment
    sample_weights = weight_lookup[moneyness_bin_indices, sqrt_t_bin_indices]

    # Improved clipping with tighter bounds and batch renormalization
    # Use more conservative clipping with softer inverse weighting
    w_min, w_max = 0.75, 1.25  # Tighter clipping bounds
    sample_weights = np.clip(sample_weights, w_min, w_max)
    
    # Batch renormalization to maintain mean = 1
    sample_weights = sample_weights / np.mean(sample_weights)

    
    print(f"Sample weights range: {sample_weights.min():.4f} - {sample_weights.max():.4f}")
    print(f"Sample weights mean (post-normalization): {sample_weights.mean():.4f}")
    print(f"Clipping bounds: [{w_min:.4f}, {w_max:.4f}] (pre-renormalization)")
    
    return sample_weights

# Vega Calculation

class VegaWeightedHuberLoss(nn.Module):
    """Custom loss function that weights Huber loss by option vega sensitivity.
    
    Emphasizes training on options with higher vega (sensitivity to volatility changes),
    which are typically at-the-money options that are most important for pricing accuracy.
    Supports weight annealing to gradually ramp up emphasis over initial epochs.
    """
    def __init__(self, delta=None):
        super(VegaWeightedHuberLoss, self).__init__()
        self.delta = delta if delta is not None else (0.05 if USE_NORMALIZED_TARGET else 1.0)
        self.annealing_factor = 1.0  # Will be updated during training
    
    def set_annealing_factor(self, factor):
        """Set the current annealing factor for weight ramping."""
        self.annealing_factor = factor
    
    def forward(self, predictions, targets, vega_weights):
        """Calculate vega-weighted Huber loss for each sample.
        
        Args:
            predictions: Model predictions [batch_size, 1]
            targets: True option prices [batch_size, 1]
            vega_weights: Vega sensitivity values [batch_size, 1]
            
        Returns:
            Per-sample loss values [batch_size] with vega weighting applied
        """

        residual = torch.abs(predictions - targets)
        
        # Huber loss: quadratic for small errors, linear for large errors
        huber_loss = torch.where(
            residual <= self.delta,
            0.5 * residual ** 2,  # Quadratic for |residual| <= delta
            self.delta * (residual - 0.5 * self.delta)  # Linear for |residual| > delta
        )
        
        # Apply ATM emphasis weighting with power 0.25 to avoid extreme weights
        vmin = 1e-3 if USE_NORMALIZED_TARGET else 0.1
        w = torch.clamp(vega_weights, min=vmin).pow(0.25)
        
        # Clip weights with tighter bounds and renormalize to maintain training stability
        w = torch.clamp(w, min=0.75, max=1.25)
        w = w / torch.mean(w)
        
        # Apply annealing: blend between uniform weights (1.0) and computed weights
        # annealing_factor = 0 means all weights = 1, annealing_factor = 1 means full weighting
        w = 1.0 + self.annealing_factor * (w - 1.0)
        
        weighted_huber = huber_loss * w
        return weighted_huber.view(-1)

# MLP model

class OptionPricingMLP(nn.Module):
    """Multi-layer perceptron for option pricing with configurable architecture.
    
    Implements a feedforward neural network with customizable layers, activation functions,
    normalization, and regularization techniques for option price prediction.
    """
    def __init__(self, input_dim: int, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = LAYER_SIZES

        layers = []
        prev = input_dim

        # Build hidden layers with configurable components
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(prev, size))
            if USE_BATCH_NORM:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.LeakyReLU(negative_slope=LEAKY_RELU_SLOPE))
            layers.append(nn.Dropout(DROPOUT_RATE))
            prev = size
        
        # Output layer with optional activation
        layers.append(nn.Linear(prev, 1))
        if OUTPUT_ACTIVATION.lower() == 'softplus':
            layers.append(nn.Softplus())
        elif OUTPUT_ACTIVATION.lower() == 'relu':
            layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using specified initialization scheme."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if USE_KAIMING_INIT:
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', 
                                           nonlinearity='leaky_relu', a=LEAKY_RELU_SLOPE)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        out = self.net(x)
        if OUTPUT_ACTIVATION.lower() in ('softplus', 'relu'):
            return out
        return out

# Comprehensive Evaluation Metrics

def calculate_comprehensive_metrics(y_true, y_pred):
    """Calculate standard regression metrics for model evaluation.
    
    Args:
        y_true: True option prices
        y_pred: Predicted option prices
        
    Returns:
        Dictionary containing MAE, MSE, and R² metrics
    """
    y_true = np.array(y_true)
    y_pred = _clip_eval_predictions(np.array(y_pred))

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {'mae': mae, 'mse': mse, 'r2': r2}

def calculate_residuals_and_errors(y_true, y_pred):
    """Calculate standardized residual and error metrics for analysis.
    
    Args:
        y_true: True option prices
        y_pred: Predicted option prices
        
    Returns:
        Tuple of (residuals, absolute_residuals, percentage_errors, absolute_percentage_errors)
    """
    y_true = np.array(y_true)
    y_pred = _clip_eval_predictions(np.array(y_pred))
    
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)
    den = np.maximum(np.abs(y_true), EPS)
    pct_error = 100 * residuals / den
    abs_pct_error = np.abs(pct_error)
    
    return residuals, abs_residuals, pct_error, abs_pct_error

def _get_input_size_for_feature_mode():
    """Calculate input size based on current FEATURE_MODE."""
    if FEATURE_MODE == 'core_only':
        return len(CORE_FEATURES)
    elif FEATURE_MODE == 'selected_features':
        return len(CORE_FEATURES) + len(SELECTED_FEATURES)
    elif FEATURE_MODE == 'all_features':
        return len(CORE_FEATURES) + len(ADDITIONAL_FEATURES)
    elif FEATURE_MODE == 'simplified_only':
        return 3  # sqrt_T, fwd_log_moneyness, historical_volatility
    elif FEATURE_MODE == 'simplified_plus_selected':
        return 3 + len(SELECTED_FEATURES)
    elif FEATURE_MODE == 'simplified_plus_all':
        return 3 + len(ADDITIONAL_FEATURES)
    else:
        return len(CORE_FEATURES)  # fallback

def _create_feature_matrix(df):
    """Create feature matrix based on FEATURE_MODE configuration.
    
    Returns:
        tuple: (X matrix, feature_names list)
    """
    # Always compute simplified features first
    S = df['spx_close'].values
    K = df['strike_price'].values
    r = df['risk_free_rate'].values
    q = df['dividend_rate'].values
    T_years = np.maximum(df['days_to_maturity'].values, 0) / DAYS_PER_YEAR
    
    # Forward price and simplified features
    F = S * np.exp((r - q) * T_years)
    fwd_log_moneyness = np.log(F / K)
    sqrt_T = np.sqrt(T_years)
    
    if FEATURE_MODE == 'core_only':
        # Only Black-Scholes core features
        X = df[CORE_FEATURES].values
        feature_names = CORE_FEATURES
        
    elif FEATURE_MODE == 'selected_features':
        # Core features + selected additional features
        core_features = df[CORE_FEATURES].values
        selected_features = df[SELECTED_FEATURES].values
        X = np.column_stack([core_features, selected_features])
        feature_names = CORE_FEATURES + SELECTED_FEATURES
        
    elif FEATURE_MODE == 'all_features':
        # Core features + all additional features
        core_features = df[CORE_FEATURES].values
        additional_features = df[ADDITIONAL_FEATURES].values
        X = np.column_stack([core_features, additional_features])
        feature_names = CORE_FEATURES + ADDITIONAL_FEATURES
        
    elif FEATURE_MODE == 'simplified_only':
        # Only simplified features + historical volatility
        hist_vol = df['historical_volatility'].values
        X = np.column_stack([sqrt_T, fwd_log_moneyness, hist_vol])
        feature_names = ['sqrt_T', 'fwd_log_moneyness', 'historical_volatility']
        
    elif FEATURE_MODE == 'simplified_plus_selected':
        # Simplified features + historical volatility + selected additional features
        hist_vol = df['historical_volatility'].values
        selected_features = df[SELECTED_FEATURES].values
        X = np.column_stack([sqrt_T, fwd_log_moneyness, hist_vol, selected_features])
        feature_names = ['sqrt_T', 'fwd_log_moneyness', 'historical_volatility'] + SELECTED_FEATURES
        
    elif FEATURE_MODE == 'simplified_plus_all':
        # Simplified features + historical volatility + all additional features
        hist_vol = df['historical_volatility'].values
        additional_features = df[ADDITIONAL_FEATURES].values
        X = np.column_stack([sqrt_T, fwd_log_moneyness, hist_vol, additional_features])
        feature_names = ['sqrt_T', 'fwd_log_moneyness', 'historical_volatility'] + ADDITIONAL_FEATURES
        
    else:
        raise ValueError(f"Invalid FEATURE_MODE: {FEATURE_MODE}. Must be one of: 'core_only', 'selected_features', 'all_features', 'simplified_only', 'simplified_plus_selected', 'simplified_plus_all'")
    
    return X, feature_names

def extract_features_from_test_data(X_test_original):
    """Extract specific features from test data for analysis."""
    n = X_test_original.shape[0]
    def get_col(name):
        idx = FEATURE_INDICES_DYNAMIC.get(name, None)
        if idx is None or idx >= X_test_original.shape[1]:
            print(f"Warning: Feature '{name}' not found or index out of bounds. Available features: {list(FEATURE_INDICES_DYNAMIC.keys())}")
            return np.zeros(n)
        return X_test_original[:, idx]
    
    # For simplified features, we need to extract from the original data
    if 'sqrt_T' in CURRENT_FEATURE_NAMES and 'fwd_log_moneyness' in CURRENT_FEATURE_NAMES:
        # If simplified features are present, we can't extract underlying components
        # Return placeholder values for compatibility
        return {
            'spx_close': get_col('spx_close') if 'spx_close' in FEATURE_INDICES_DYNAMIC else np.ones(n),
            'strike_price': get_col('strike_price') if 'strike_price' in FEATURE_INDICES_DYNAMIC else np.ones(n),
            'days_to_maturity': get_col('days_to_maturity') if 'days_to_maturity' in FEATURE_INDICES_DYNAMIC else np.ones(n),
            'risk_free_rate': get_col('risk_free_rate') if 'risk_free_rate' in FEATURE_INDICES_DYNAMIC else np.zeros(n),
            'dividend_rate': get_col('dividend_rate') if 'dividend_rate' in FEATURE_INDICES_DYNAMIC else np.zeros(n),
            'historical_volatility': get_col('historical_volatility') if 'historical_volatility' in FEATURE_INDICES_DYNAMIC else np.ones(n) * 0.2
        }
    else:
        return {
            'spx_close': get_col('spx_close'),
            'strike_price': get_col('strike_price'),
            'days_to_maturity': get_col('days_to_maturity'),
            'risk_free_rate': get_col('risk_free_rate'),
            'dividend_rate': get_col('dividend_rate'),
            'historical_volatility': get_col('historical_volatility')
        }


def create_feature_analysis_subplot(ax, df_test, feature_buckets, bucket_labels, error_column, title, xlabel, ylabel, color, rotation=0):
    """Helper function to create a single feature analysis subplot"""
    values = []
    counts = []
    actual_labels = []
    
    # Handle dynamic label generation for quantile-based buckets
    if bucket_labels is None:
        # For quantile buckets, generate labels with actual bounds
        if hasattr(feature_buckets, 'cat'):
            # Categorical buckets
            if hasattr(feature_buckets, 'cat'):
                categories = feature_buckets.cat.categories
            else:
                categories = feature_buckets.categories
            for i, bucket in enumerate(categories):
                mask = feature_buckets == bucket
                if mask.sum() > 0:
                    values.append(np.mean(df_test[mask][error_column]))
                    counts.append(mask.sum())
                    actual_labels.append(str(bucket))
        else:
            # Numeric buckets - generate bounds labels
            unique_buckets = sorted(feature_buckets.dropna().unique())
            for bucket_val in unique_buckets:
                mask = feature_buckets == bucket_val
                if mask.sum() > 0:
                    subset_data = df_test[mask]
                    if 'mid_price' in df_test.columns and 'price' in title.lower():
                        # Price quantiles (for both MAE and MAPE)
                        min_val = subset_data['mid_price'].min()
                        max_val = subset_data['mid_price'].max()
                        actual_labels.append(f'${min_val:.0f}-\n${max_val:.0f}')
                    elif 'historical_volatility' in df_test.columns and 'volatility' in title.lower():
                        # Volatility quantiles
                        min_val = subset_data['historical_volatility'].min()
                        max_val = subset_data['historical_volatility'].max()
                        actual_labels.append(f'{min_val:.2f}-\n{max_val:.2f}')
                    else:
                        actual_labels.append(f'Q{bucket_val+1}')
                    values.append(np.mean(subset_data[error_column]))
                    counts.append(mask.sum())
    else:
        # Use provided labels
        # Check if it's a pandas Series with categorical dtype or direct Categorical
        if hasattr(feature_buckets, 'cat'):
            categories = feature_buckets.cat.categories
        else:
            categories = feature_buckets.categories
        
        for i, bucket in enumerate(categories):
            mask = feature_buckets == bucket
            if mask.sum() > 0:
                values.append(np.mean(df_test[mask][error_column]))
                counts.append(mask.sum())
                actual_labels.append(bucket_labels[i] if i < len(bucket_labels) else str(bucket))
    
    if not values:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold')
        return
    
    bars = ax.bar(actual_labels, values, color=color, alpha=0.8, edgecolor='black')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    if rotation > 0:
        ax.tick_params(axis='x', rotation=rotation)
    
    # Add count annotations
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'n={count:,}', ha='center', va='bottom', fontsize=9)

def create_mae_analysis(df_test, results_dir):
    """Create comprehensive MAE analysis plots by various features"""
    print("Creating MAE analysis plots...")
    
    # Calculate absolute errors
    df_test['absolute_error'] = np.abs(df_test['mid_price'] - df_test['predicted_price'])
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Mean Absolute Error (MAE) Analysis by Features', fontsize=16, fontweight='bold', y=1.02)
    
    # Define all feature configurations
    feature_configs = [
        {
            'ax': axes[0, 0],
            'buckets': pd.qcut(df_test['mid_price'], q=5, labels=False),
            'labels': None,  # Will be computed dynamically with bounds
            'title': 'MAE by Price Quantiles',
            'xlabel': 'Price Quantile',
            'color': 'lightcoral'
        },
        {
            'ax': axes[0, 1],
            'buckets': pd.cut(df_test['moneyness'], bins=[0, 0.9, 1.1, np.inf], labels=['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)'], include_lowest=True),
            'labels': ['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)'],
            'title': 'MAE by Moneyness',
            'xlabel': 'Moneyness',
            'color': 'lightblue'
        },
        {
            'ax': axes[1, 0],
            'buckets': pd.cut(df_test['days_to_maturity'] / DAYS_PER_YEAR, bins=[0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf], labels=['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)'], include_lowest=True),
            'labels': ['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)'],
            'title': 'MAE by Time to Expiration',
            'xlabel': 'Time to Maturity',
            'color': 'lightgreen'
        },
        {
            'ax': axes[1, 1],
            'buckets': pd.qcut(df_test['historical_volatility'], q=5, labels=False) if 'historical_volatility' in df_test.columns else None,
            'labels': None,  # Will be computed dynamically with bounds
            'title': 'MAE by Historical Volatility Quintiles',
            'xlabel': 'Historical Volatility Quintile',
            'color': 'gold'
        },
        {
            'ax': axes[2, 0],
            'buckets': pd.cut(df_test['volume'], bins=[0, 1, 100, 1000, np.inf], labels=['Zero', 'Low (1-100)', 'Medium (101-1000)', 'High (1001+)'], include_lowest=True),
            'labels': ['Zero', 'Low (1-100)', 'Medium (101-1000)', 'High (1001+)'],
            'title': 'MAE by Volume',
            'xlabel': 'Volume',
            'color': 'purple',
            'rotation': 45
        },
        {
            'ax': axes[2, 1],
            'buckets': pd.cut(df_test['equity_uncertainty'], bins=[0, 50, 100, 150, np.inf], labels=['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)'], include_lowest=True) if 'equity_uncertainty' in df_test.columns else None,
            'labels': ['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)'],
            'title': 'MAE by Equity Uncertainty',
            'xlabel': 'Equity Uncertainty',
            'color': 'orange',
            'rotation': 45
        }
    ]
    
    # Create all subplots
    for config in feature_configs:
        if config['buckets'] is not None:
            create_feature_analysis_subplot(
                config['ax'], df_test, config['buckets'], config['labels'],
                'absolute_error', config['title'], config['xlabel'],
                'Mean Absolute Error ($)', config['color'],
                config.get('rotation', 0)
            )
        else:
            config['ax'].text(0.5, 0.5, 'Feature not available\nin current mode', 
                            ha='center', va='center', transform=config['ax'].transAxes)
            config['ax'].set_title(config['title'], fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'mae_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAE analysis plot saved to {save_path}")

def create_mape_analysis(df_test, results_dir):
    """Create comprehensive MAPE analysis plots by various features"""
    print("Creating MAPE analysis plots...")
    
    # Calculate percentage errors
    df_test['percentage_error'] = (np.abs(df_test['mid_price'] - df_test['predicted_price']) / 
                                  np.maximum(df_test['mid_price'], 1.0)) * 100
    
    # Create figure with 3x2 subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Mean Absolute Percentage Error (MAPE) Analysis by Features', fontsize=16, fontweight='bold', y=1.02)
    
    # Define all feature configurations (same as MAE but with percentage_error)
    feature_configs = [
        {
            'ax': axes[0, 0],
            'buckets': pd.qcut(df_test['mid_price'], q=5, labels=False),
            'labels': None,  # Will be computed dynamically with bounds
            'title': 'MAPE by Price Quantiles',
            'xlabel': 'Price Quantile',
            'color': 'lightcoral'
        },
        {
            'ax': axes[0, 1],
            'buckets': pd.cut(df_test['moneyness'], bins=[0, 0.9, 1.1, np.inf], labels=['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)'], include_lowest=True),
            'labels': ['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)'],
            'title': 'MAPE by Moneyness',
            'xlabel': 'Moneyness',
            'color': 'lightblue'
        },
        {
            'ax': axes[1, 0],
            'buckets': pd.cut(df_test['days_to_maturity'] / DAYS_PER_YEAR, bins=[0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf], labels=['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)'], include_lowest=True),
            'labels': ['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)'],
            'title': 'MAPE by Time to Expiration',
            'xlabel': 'Time to Maturity',
            'color': 'lightgreen'
        },
        {
            'ax': axes[1, 1],
            'buckets': pd.qcut(df_test['historical_volatility'], q=5, labels=False) if 'historical_volatility' in df_test.columns else None,
            'labels': None,  # Will be computed dynamically with bounds
            'title': 'MAPE by Historical Volatility Quintiles',
            'xlabel': 'Historical Volatility Quintile',
            'color': 'gold'
        },
        {
            'ax': axes[2, 0],
            'buckets': pd.cut(df_test['volume'], bins=[0, 1, 100, 1000, np.inf], labels=['Zero', 'Low (1-100)', 'Medium (101-1000)', 'High (1001+)'], include_lowest=True),
            'labels': ['Zero', 'Low (1-100)', 'Medium (101-1000)', 'High (1001+)'],
            'title': 'MAPE by Volume',
            'xlabel': 'Volume',
            'color': 'purple',
            'rotation': 45
        },
        {
            'ax': axes[2, 1],
            'buckets': pd.cut(df_test['equity_uncertainty'], bins=[0, 50, 100, 150, np.inf], labels=['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)'], include_lowest=True) if 'equity_uncertainty' in df_test.columns else None,
            'labels': ['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)'],
            'title': 'MAPE by Equity Uncertainty',
            'xlabel': 'Equity Uncertainty',
            'color': 'orange',
            'rotation': 45
        }
    ]
    
    # Create all subplots
    for config in feature_configs:
        if config['buckets'] is not None:
            create_feature_analysis_subplot(
                config['ax'], df_test, config['buckets'], config['labels'],
                'percentage_error', config['title'], config['xlabel'],
                'Mean Absolute Percentage Error (%)', config['color'],
                config.get('rotation', 0)
            )
        else:
            config['ax'].text(0.5, 0.5, 'Feature not available\nin current mode', 
                            ha='center', va='center', transform=config['ax'].transAxes)
            config['ax'].set_title(config['title'], fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(results_dir, 'mape_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAPE analysis plot saved to {save_path}")

def write_comprehensive_diagnostics(df_test, results_dir):
    """Write comprehensive diagnostic statistics by features matching Black-Scholes structure"""
    print("Writing comprehensive diagnostic statistics...")
    
    # Calculate errors
    df_test['absolute_error'] = np.abs(df_test['mid_price'] - df_test['predicted_price'])
    df_test['percentage_error'] = (np.abs(df_test['mid_price'] - df_test['predicted_price']) / 
                                  np.maximum(df_test['mid_price'], 1.0)) * 100
    
    diagnostics_path = os.path.join(results_dir, 'comprehensive_diagnostics.txt')
    
    with open(diagnostics_path, 'w') as f:
        f.write("=== COMPREHENSIVE DIAGNOSTIC STATISTICS ===\n\n")
        
        # Overall Statistics
        f.write("--- OVERALL PERFORMANCE ---\n")
        f.write(f"Total Test Samples: {len(df_test):,}\n")
        f.write(f"Mean Absolute Error: ${df_test['absolute_error'].mean():.2f}\n")
        f.write(f"Mean Absolute Percentage Error: {df_test['percentage_error'].mean():.2f}%\n")
        f.write(f"Median Absolute Error: ${df_test['absolute_error'].median():.2f}\n")
        f.write(f"Median Absolute Percentage Error: {df_test['percentage_error'].median():.2f}%\n")
        f.write(f"Standard Deviation of Errors: ${df_test['absolute_error'].std():.2f}\n\n")
        
        # Statistics by Price Quantiles
        f.write("--- STATISTICS BY PRICE QUANTILES ---\n")
        price_quantiles = pd.qcut(df_test['mid_price'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            mask = price_quantiles == q
            if mask.sum() > 0:
                subset = df_test[mask]
                price_range = f"${subset['mid_price'].min():.2f} - ${subset['mid_price'].max():.2f}"
                f.write(f"{q} ({price_range}):\n")
                f.write(f"  Count: {mask.sum():,}\n")
                f.write(f"  MAE: ${subset['absolute_error'].mean():.2f}\n")
                f.write(f"  MAPE: {subset['percentage_error'].mean():.2f}%\n")
                f.write(f"  Median AE: ${subset['absolute_error'].median():.2f}\n")
                f.write(f"  Std AE: ${subset['absolute_error'].std():.2f}\n\n")
        
        # Statistics by Moneyness
        f.write("--- STATISTICS BY MONEYNESS ---\n")
        # Use pd.cut directly like Black-Scholes
        MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
        MONEYNESS_BIN_LABELS = ['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)']
        df_test['moneyness_bucket'] = pd.cut(df_test['moneyness'], bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)
        
        moneyness_stats = df_test.groupby('moneyness_bucket').agg({
            'absolute_error': ['count', 'mean', 'median', 'std'],
            'percentage_error': 'mean'
        }).round(2)
        
        for bucket in moneyness_stats.index:
            if pd.isna(bucket):
                continue
            count = int(moneyness_stats.loc[bucket, ('absolute_error', 'count')])
            mae = moneyness_stats.loc[bucket, ('absolute_error', 'mean')]
            mape = moneyness_stats.loc[bucket, ('percentage_error', 'mean')]
            median_ae = moneyness_stats.loc[bucket, ('absolute_error', 'median')]
            std_ae = moneyness_stats.loc[bucket, ('absolute_error', 'std')]
            
            f.write(f"{bucket}:\n")
            f.write(f"  Count: {count:,}\n")
            f.write(f"  MAE: ${mae:.2f}\n")
            f.write(f"  MAPE: {mape:.2f}%\n")
            f.write(f"  Median AE: ${median_ae:.2f}\n")
            f.write(f"  Std AE: ${std_ae:.2f}\n\n")
        
        # Statistics by Time to Maturity
        f.write("--- STATISTICS BY TIME TO MATURITY ---\n")
        # Use pd.cut directly like Black-Scholes
        TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]
        TIME_BIN_LABELS = ['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)']
        df_test['time_bucket'] = pd.cut(df_test['days_to_maturity'] / DAYS_PER_YEAR, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)
        
        time_stats = df_test.groupby('time_bucket').agg({
            'absolute_error': ['count', 'mean', 'median', 'std'],
            'percentage_error': 'mean'
        }).round(2)
        
        for bucket in time_stats.index:
            if pd.isna(bucket):
                continue
            count = int(time_stats.loc[bucket, ('absolute_error', 'count')])
            mae = time_stats.loc[bucket, ('absolute_error', 'mean')]
            mape = time_stats.loc[bucket, ('percentage_error', 'mean')]
            median_ae = time_stats.loc[bucket, ('absolute_error', 'median')]
            std_ae = time_stats.loc[bucket, ('absolute_error', 'std')]
            
            f.write(f"{bucket}:\n")
            f.write(f"  Count: {count:,}\n")
            f.write(f"  MAE: ${mae:.2f}\n")
            f.write(f"  MAPE: {mape:.2f}%\n")
            f.write(f"  Median AE: ${median_ae:.2f}\n")
            f.write(f"  Std AE: ${std_ae:.2f}\n\n")
        
        # Statistics by Historical Volatility
        if 'historical_volatility' in df_test.columns:
            f.write("--- STATISTICS BY HISTORICAL VOLATILITY ---\n")
            hist_vol_quintiles = pd.qcut(df_test['historical_volatility'], q=5, labels=False, duplicates='drop')
            unique_quintiles = sorted(hist_vol_quintiles[~pd.isna(hist_vol_quintiles)].unique())
            for q in unique_quintiles:
                mask = hist_vol_quintiles == q
                if mask.sum() > 0:
                    subset = df_test[mask]
                    vol_range = f"{subset['historical_volatility'].min():.3f} - {subset['historical_volatility'].max():.3f}"
                    f.write(f"Q{q+1} ({vol_range}):\n")
                    f.write(f"  Count: {mask.sum():,}\n")
                    f.write(f"  MAE: ${subset['absolute_error'].mean():.2f}\n")
                    f.write(f"  MAPE: {subset['percentage_error'].mean():.2f}%\n")
                    f.write(f"  Median AE: ${subset['absolute_error'].median():.2f}\n")
                    f.write(f"  Std AE: ${subset['absolute_error'].std():.2f}\n\n")
        else:
            f.write("--- STATISTICS BY HISTORICAL VOLATILITY ---\n")
            f.write("Historical volatility not available in current feature set.\n\n")
        
        # Statistics by Volume using direct pd.cut approach
        f.write("--- STATISTICS BY VOLUME ---\n")
        df_test['volume_bucket'] = pd.cut(df_test['volume'], bins=[-0.1, 0.5, 100.5, 1000.5, np.inf], 
                                         labels=['Zero', 'Low (1-100)', 'Medium (101-1000)', 'High (1001+)'], include_lowest=True)
        volume_stats = df_test.groupby('volume_bucket').agg({
            'absolute_error': ['mean', 'median', 'std', 'count'],
            'percentage_error': 'mean'
        }).round(2)
        
        for bucket in volume_stats.index:
            if pd.isna(bucket):
                continue
            count = int(volume_stats.loc[bucket, ('absolute_error', 'count')])
            mae = volume_stats.loc[bucket, ('absolute_error', 'mean')]
            mape = volume_stats.loc[bucket, ('percentage_error', 'mean')]
            median_ae = volume_stats.loc[bucket, ('absolute_error', 'median')]
            std_ae = volume_stats.loc[bucket, ('absolute_error', 'std')]
            
            f.write(f"{bucket}:\n")
            f.write(f"  Count: {count:,}\n")
            f.write(f"  MAE: ${mae:.2f}\n")
            f.write(f"  MAPE: {mape:.2f}%\n")
            f.write(f"  Median AE: ${median_ae:.2f}\n")
            f.write(f"  Std AE: ${std_ae:.2f}\n\n")
        
        # Statistics by Equity Uncertainty using direct pd.cut approach
        if 'equity_uncertainty' in df_test.columns:
            f.write("--- STATISTICS BY EQUITY UNCERTAINTY ---\n")
            df_test['uncertainty_bucket'] = pd.cut(df_test['equity_uncertainty'], bins=[0, 50, 100, 150, np.inf], 
                                                  labels=['Low (<50)', 'Medium (50-100)', 'High (100-150)', 'Very High (>150)'], include_lowest=True)
            uncertainty_stats = df_test.groupby('uncertainty_bucket').agg({
                'absolute_error': ['mean', 'median', 'std', 'count'],
                'percentage_error': 'mean'
            }).round(2)
            
            for bucket in uncertainty_stats.index:
                if pd.isna(bucket):
                    continue
                count = int(uncertainty_stats.loc[bucket, ('absolute_error', 'count')])
                mae = uncertainty_stats.loc[bucket, ('absolute_error', 'mean')]
                mape = uncertainty_stats.loc[bucket, ('percentage_error', 'mean')]
                median_ae = uncertainty_stats.loc[bucket, ('absolute_error', 'median')]
                std_ae = uncertainty_stats.loc[bucket, ('absolute_error', 'std')]
                
                f.write(f"{bucket}:\n")
                f.write(f"  Count: {count:,}\n")
                f.write(f"  MAE: ${mae:.2f}\n")
                f.write(f"  MAPE: {mape:.2f}%\n")
                f.write(f"  Median AE: ${median_ae:.2f}\n")
                f.write(f"  Std AE: ${std_ae:.2f}\n\n")
        else:
            f.write("--- STATISTICS BY EQUITY UNCERTAINTY ---\n")
            f.write("Equity uncertainty not available in current feature set.\n\n")
    
    print(f"Comprehensive diagnostics saved to {diagnostics_path}")


def calculate_bucketed_metrics(y_true, y_pred, df_test_or_X_original, results_dir):
    """
    Calculate and save detailed bucketed metrics by moneyness and time to maturity.
    This is critical for normalized targets as raw dollar evaluation can mask improvements.
    """
    y_pred = _clip_eval_predictions(y_pred)
    residuals, abs_residuals, pct_error, abs_pct_error = calculate_residuals_and_errors(y_true, y_pred)
    
    # Handle both dataframe and matrix inputs
    if isinstance(df_test_or_X_original, pd.DataFrame):
        df_test = df_test_or_X_original
        spx_close = df_test['spx_close'].values if 'spx_close' in df_test.columns else np.full(len(df_test), np.nan)
        strike_price = df_test['strike_price'].values if 'strike_price' in df_test.columns else np.full(len(df_test), np.nan)
        days_to_maturity = df_test['days_to_maturity'].values if 'days_to_maturity' in df_test.columns else np.full(len(df_test), np.nan)
    else:
        # Legacy matrix input - use centralized feature extraction
        features = extract_features_from_test_data(df_test_or_X_original)
        spx_close = features['spx_close']
        strike_price = features['strike_price']
        days_to_maturity = features['days_to_maturity']
    
    # Calculate moneyness and time to expiration
    moneyness = spx_close / strike_price
    time_to_expiration = days_to_maturity / DAYS_PER_YEAR  # Convert to years
    
    # Use pd.cut directly like Black-Scholes (consistent with other functions)
    MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
    MONEYNESS_BIN_LABELS = ['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)']
    TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]
    TIME_BIN_LABELS = ['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)']
    
    moneyness_buckets = pd.cut(moneyness, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)
    time_buckets = pd.cut(time_to_expiration, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)
    
    # Create DataFrame for analysis using groupby approach (avoids categorical iteration issues)
    analysis_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'abs_residual': abs_residuals,
        'abs_pct_error': abs_pct_error,
        'moneyness_bucket': moneyness_buckets,
        'time_bucket': time_buckets
    })
    
    # Calculate metrics by moneyness buckets using groupby
    print(f"\n--- Bucketed Metrics by Moneyness ---")
    moneyness_analysis = analysis_df.groupby('moneyness_bucket').agg({
        'abs_residual': 'mean',
        'abs_pct_error': 'mean',
        'y_true': 'count'
    }).reset_index()
    moneyness_analysis.columns = ['bucket', 'mae', 'mape', 'count']
    
    moneyness_metrics = {}
    for _, row in moneyness_analysis.iterrows():
        bucket = row['bucket']
        moneyness_metrics[bucket] = {
            'mae': row['mae'],
            'mape': row['mape'],
            'count': row['count']
        }
        print(f"  {bucket}: MAE=${row['mae']:.2f}, MAPE={row['mape']:.1f}%, n={row['count']:,}")

    # Calculate metrics by time buckets using groupby
    print(f"\n--- Bucketed Metrics by Time to Maturity ---")
    time_analysis = analysis_df.groupby('time_bucket').agg({
        'abs_residual': 'mean',
        'abs_pct_error': 'mean',
        'y_true': 'count'
    }).reset_index()
    time_analysis.columns = ['bucket', 'mae', 'mape', 'count']
    
    time_metrics = {}
    for _, row in time_analysis.iterrows():
        bucket = row['bucket']
        time_metrics[bucket] = {
            'mae': row['mae'],
            'mape': row['mape'],
            'count': row['count']
        }
        print(f"  {bucket}: MAE=${row['mae']:.2f}, MAPE={row['mape']:.1f}%, n={row['count']:,}")
    
    # Save detailed bucketed metrics to CSV
    bucketed_results_path = os.path.join(results_dir, 'bucketed_metrics.csv')
    
    # Create detailed DataFrame for analysis
    df_bucketed = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'residual': residuals,
        'abs_residual': abs_residuals,
        'pct_error': pct_error,
        'abs_pct_error': abs_pct_error,
        'moneyness': moneyness,
        'time_to_expiration': time_to_expiration,
        'moneyness_bucket': moneyness_buckets,
        'time_bucket': time_buckets
    })
    
    df_bucketed.to_csv(bucketed_results_path, index=False)
    print(f"Detailed bucketed metrics saved to: {bucketed_results_path}")
    
    return moneyness_metrics, time_metrics

# Time-based and Improved Data Splitting

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

# Multi-dimensional Stratification Helper

def _build_multidim_strata(df_sub: pd.DataFrame) -> np.ndarray:
    """
    Build combined stratification labels: price-quantile × moneyness × time buckets.
    Falls back gracefully if required columns are missing or strata are too sparse.
    Returns an array of labels suitable for sklearn's stratify= argument.
    """
    required = {'mid_price', 'spx_close', 'strike_price', 'days_to_maturity'}
    if not required.issubset(df_sub.columns):
        # Fallback: price-only quantiles
        return pd.qcut(df_sub['mid_price'].values, q=5, labels=False, duplicates='drop')

    # Compute components
    price_q = pd.qcut(df_sub['mid_price'].values, q=5, labels=False, duplicates='drop')
    mny = (df_sub['spx_close'].values / np.maximum(df_sub['strike_price'].values, 1e-12))
    t_years = np.maximum(df_sub['days_to_maturity'].values, 0) / DAYS_PER_YEAR

    # Moneyness buckets (centralized)
    mny_bucket = pd.cut(mny, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS)

    # Time buckets (centralized)
    t_bucket = pd.cut(t_years, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS)

    # Combine
    combo = pd.Series(price_q).astype(str) + '_' + mny_bucket.astype(str) + '_' + t_bucket.astype(str)

    # Guard against sparsity
    counts = combo.value_counts()
    rare = counts[counts < 10].index
    combo = combo.mask(combo.isin(rare), other='OTHER')

    # If too many collapsed to OTHER, fallback to price-only quantiles
    if (combo == 'OTHER').mean() > 0.8:
        return price_q

    return np.asarray(combo.values)

def create_random_splits(X, y, indices, df_original=None):
    """
    Create random splits using EXACT same logic as Black-Scholes script.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test
    """
    
    # Create data view for splitting
    df_view = df_original.iloc[indices].reset_index(drop=True)
    
    # Decide stratify values
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

# Data loading & preprocessing
ALL_FEATURES = CORE_FEATURES + ADDITIONAL_FEATURES

TARGET_COLUMN = "mid_price"

# Dynamic feature order and indices derived from the actual design matrix used.
# This eliminates reliance on assumed ordering and keeps indices aligned with X matrices.
CURRENT_FEATURE_NAMES = list(ALL_FEATURES)  # will be updated where X is constructed
FEATURE_INDICES_DYNAMIC = {name: i for i, name in enumerate(CURRENT_FEATURE_NAMES)}

def validate_data_quality(df: pd.DataFrame):
    """Validate data quality before training and return a list of human-readable issues.
    Defensive against missing columns.
    """
    issues = []

    def has(cols):
        return all(c in df.columns for c in cols)

    # 1) Basic price sanity
    if 'mid_price' in df.columns:
        n_nonpos = int((df['mid_price'] <= 0).sum())
        if n_nonpos > 0:
            issues.append(f"{n_nonpos} rows with negative or zero mid_price")

    # 2) Extreme moneyness
    if has(['spx_close', 'strike_price']):
        mny = df['spx_close'] / df['strike_price'].replace(0, np.nan)
        n_extreme_hi = int((mny > 5).sum())
        n_extreme_lo = int((mny < 0.2).sum())
        if n_extreme_hi + n_extreme_lo > 0:
            issues.append(f"{n_extreme_hi + n_extreme_lo} rows with extreme moneyness (S/K < 0.2 or > 5)")

    # 3) Negative time to maturity
    if 'days_to_maturity' in df.columns:
        n_neg_t = int((df['days_to_maturity'] < 0).sum())
        if n_neg_t > 0:
            issues.append(f"{n_neg_t} rows with negative days_to_maturity")

    # 4) NaNs in key fields
    key_cols = [c for c in ['mid_price','best_bid','best_offer','spx_close','strike_price','days_to_maturity'] if c in df.columns]
    if key_cols:
        n_nans = int(df[key_cols].isna().any(axis=1).sum())
        if n_nans > 0:
            issues.append(f"{n_nans} rows with NaNs in key columns {key_cols}")

    return issues

def load_dataset(csv_path: str, sample_size: int = 1000000):
    
    df = pd.read_csv(csv_path)
    
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=SEED)
        print(f"Sampled {sample_size} rows from {len(df)} available")
    # Ensure required columns for advanced training features are loaded
    required_columns = set(ALL_FEATURES + [TARGET_COLUMN])
    
    # Note: bs_vega is computed from scratch using historical volatility, not loaded from dataset
    
    # Check if all required columns exist
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Warning: Missing columns in dataset: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
    
    # Convert date column if it exists for time-based splitting
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
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
    if 'moneyness' in df.columns and (MIN_MONEYNESS is not None or MAX_MONEYNESS is not None):
        initial_count = len(df)
        if MIN_MONEYNESS is not None and MAX_MONEYNESS is not None:
            df = df[(df['moneyness'] >= MIN_MONEYNESS) & (df['moneyness'] <= MAX_MONEYNESS)]
            print(f"Applied moneyness filter ({MIN_MONEYNESS} ≤ S/K ≤ {MAX_MONEYNESS}): {len(df):,} rows (removed {initial_count - len(df):,})")
        
        elif MIN_MONEYNESS is not None:
            df = df[df['moneyness'] >= MIN_MONEYNESS]
            print(f"Applied moneyness filter (S/K ≥ {MIN_MONEYNESS}): {len(df):,} rows (removed {initial_count - len(df):,})")
        
        elif MAX_MONEYNESS is not None:
            df = df[df['moneyness'] <= MAX_MONEYNESS]
            print(f"Applied moneyness filter (S/K ≤ {MAX_MONEYNESS}): {len(df):,} rows (removed {initial_count - len(df):,})")
    
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
    
    # Apply quality filters for high-quality options data
    initial_count = len(df)
    
    # Filter 1: Ask > Bid (valid bid-ask spread)
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
        initial_count = len(df)
        spread_ratio = (df['best_offer'] - df['best_bid']) / df['mid_price']
        df = df[spread_ratio <= MAX_SPREAD_RATIO]
        print(f"Removed {initial_count - len(df)} options with bid-ask spread ratio > {MAX_SPREAD_RATIO*100:.0f}% (legacy)")
    
    # Filter 4: Removed - keeping options with zero open interest as requested
    
    # Step 0: Remove rows with missing values in key columns
    initial_count = len(df)
    existing_required_cols = [col for col in ALL_FEATURES + [TARGET_COLUMN] if col in df.columns]
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
    
    # Zero volume filtering
    if 'volume' in df.columns and ZERO_VOLUME_INCLUSION < 1.0:
        initial_count = len(df)
        zero_volume_mask = df['volume'] == 0
        non_zero_volume_df = df[~zero_volume_mask]
        zero_volume_df = df[zero_volume_mask]
        
        print(f"Zero volume options: {len(zero_volume_df):,} rows")
        print(f"Non-zero volume options: {len(non_zero_volume_df):,} rows")
        
        if ZERO_VOLUME_INCLUSION > 0.0 and len(zero_volume_df) > 0:
            # Sample the specified proportion of zero-volume options
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
        
        if USE_TIME_BASED_SPLIT and USE_TIME_PERCENT_SPLIT and 'date' in df.columns:
            df = df.sort_values('date').reset_index(drop=True)
            print("Re-sorted dataset by date to support time-based percent split.")
        
        else:
            df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
            print("Shuffled dataset rows to avoid ordering bias during random split.")

    issues = validate_data_quality(df)
    if issues:
        print("\nData quality issues detected:")
        for msg in issues:
            print(" -", msg)
        DATA_QUALITY_ISSUES.extend(issues)
    
    print(f"Final dataset size: {len(df)}")
    
    # Print data range diagnostics
    print(f"\n--- Data Range Diagnostics ---")
    print(f"Mid Price Range: ${df[TARGET_COLUMN].min():.2f} - ${df[TARGET_COLUMN].max():.2f}")
    print(f"Mid Price Mean: ${df[TARGET_COLUMN].mean():.2f}")
    print(f"Mid Price Median: ${df[TARGET_COLUMN].median():.2f}")
    print(f"Mid Price 95th percentile: ${df[TARGET_COLUMN].quantile(0.95):.2f}")
    print(f"Mid Price 99th percentile: ${df[TARGET_COLUMN].quantile(0.99):.2f}")
    print(f"Strike Price Range: ${df['strike_price'].min():.2f} - ${df['strike_price'].max():.2f}")
    print(f"SPX Close Range: ${df['spx_close'].min():.2f} - ${df['spx_close'].max():.2f}")
    
    # Count options by price ranges using cleaner approach
    price_bins = pd.cut(df[TARGET_COLUMN], 
                       bins=[0, 10, 50, 100, 200, 500, 1000, float('inf')],
                       labels=['<$10', '$10-50', '$50-100', '$100-200', '$200-500', '$500-1000', '>$1000'])
    
    print(f"\nOptions by Price Range:")
    if hasattr(price_bins, 'cat'):
        price_categories = price_bins.cat.categories
    else:
        price_categories = price_bins.categories
    
    for label in price_categories:
        count = (price_bins == label).sum()
        pct = count / len(df) * 100
        print(f"  {label}: {count:,} options ({pct:.1f}%)")

    # Filter out NaN/missing values for advanced training features
    print(f"\n--- Filtering NaN Values for Advanced Training Features ---")
    initial_size = len(df)
    
    # Moneyness × √T weighting uses standard market data (no NaN filtering needed beyond basic columns)
    
    # Report final filtering results
    final_size = len(df)
    if final_size < initial_size:
        removed_count = initial_size - final_size
        print(f"Total rows removed due to NaN values: {removed_count:,}")
        print(f"Final dataset size after NaN filtering: {final_size:,}")

    # Check if dataset is empty after filtering
    if len(df) == 0:
        raise ValueError("ERROR: No data remaining after filtering! All rows were removed by the filtering pipeline. Please check your filtering parameters.")

    # Extract features and target based on FEATURE_MODE
    X, feature_names = _create_feature_matrix(df)
    
    # Update dynamic feature mapping
    global CURRENT_FEATURE_NAMES, FEATURE_INDICES_DYNAMIC
    CURRENT_FEATURE_NAMES = feature_names
    FEATURE_INDICES_DYNAMIC = {name: i for i, name in enumerate(CURRENT_FEATURE_NAMES)}
    
    print(f"\n--- Feature Configuration: {FEATURE_MODE.upper()} ---")
    print(f"Total features used: {len(feature_names)}")
    print(f"Features: {feature_names}")
    
    y = df[TARGET_COLUMN].values
    
    # Handle normalized target (price/S) if enabled
    if USE_NORMALIZED_TARGET:
        print(f"\n--- Using Normalized Target (price/S) ---")
        underlying_prices = df['spx_close'].values 
        y_normalized = y / underlying_prices
        print(f"Original price range: ${y.min():.2f} - ${y.max():.2f}")
        print(f"Normalized target range: {y_normalized.min():.4f} - {y_normalized.max():.4f}")
        y = y_normalized

    # Extract vega weights if vega-weighted loss is enabled
    vega_weights = None
    if USE_VEGA_WEIGHTED_LOSS:
        print(f"\n--- Using Vega-Weighted Loss ---")
        # Compute leak-free vega using historical volatility (bs_vega_safe)
        if all(col in df.columns for col in ['spx_close', 'strike_price', 'risk_free_rate', 'dividend_rate', 'days_to_maturity', 'historical_volatility']):
            print("Computing leak-free vega (bs_vega_safe) using historical volatility...")
            
            # Calculate vega using Black-Scholes formula
            S = df['spx_close'].values
            K = df['strike_price'].values
            r = df['risk_free_rate'].values
            q = df['dividend_rate'].values
            T = df['days_to_maturity'].values / DAYS_PER_YEAR
            sigma = df['historical_volatility'].values
            
            vega_weights = calculate_black_scholes_vega(S, K, r, q, T, sigma)
            
            print(f"Raw leak-free vega range: {vega_weights.min():.4f} - {vega_weights.max():.4f}")
            print(f"Raw leak-free vega mean: {vega_weights.mean():.4f}")
        else:
            print("Error: Cannot compute vega weights. Missing required columns for vega calculation.")
            print("Required columns: spx_close, strike_price, risk_free_rate, dividend_rate, days_to_maturity, historical_volatility")
            print("Available columns:", df.columns.tolist())
            vega_weights = None

    # Compute Moneyness × √T sample weights if enabled
    # Sample weights will be computed after splitting to prevent data leakage
    return X, y, df, vega_weights

def export_predictions_csv(df_original, test_indices, X_test_original, y_test_original, predictions, results_dir):
    """Export CSV with date, inputs, actual prices, predictions, errors, and % errors"""
    print("\n=== Exporting Predictions CSV ===")
    
    # Get the test data from original dataframe
    df_test = df_original.iloc[test_indices].copy()
    
    # Add original feature values (unscaled) - handle simplified features
    for i, feature in enumerate(CURRENT_FEATURE_NAMES):
        df_test[f'{feature}_input'] = X_test_original[:, i]
    
    # Add actual and predicted prices
    df_test['actual_mid_price'] = y_test_original
    df_test['predicted_mid_price'] = predictions
    
    # Calculate errors
    df_test['absolute_error'] = np.abs(y_test_original - predictions)
    # Use denominator floor to prevent explosion on tiny prices
    df_test['percentage_error'] = (np.abs(y_test_original - predictions) / np.maximum(y_test_original, 1.0)) * 100
    
    # Select and reorder columns for export
    export_columns = ['date'] if 'date' in df_test.columns else []
    
    # Add input features
    for feature in ALL_FEATURES:
        export_columns.append(f'{feature}_input')
    
    # Add prices and errors
    export_columns.extend([
        'actual_mid_price', 'predicted_mid_price', 
        'absolute_error', 'percentage_error'
    ])
    
    # Filter columns that exist in the dataframe
    available_columns = [col for col in export_columns if col in df_test.columns]
    
    # If no date column, add index as identifier
    if 'date' not in df_test.columns:
        df_test['sample_id'] = range(len(df_test))
        available_columns = ['sample_id'] + available_columns
    
    # Export to CSV
    csv_path = os.path.join(results_dir, 'mlp_predictions.csv')
    df_test[available_columns].to_csv(csv_path, index=False)
    
    print(f"Predictions CSV exported to: {csv_path}")
    print(f"CSV contains {len(df_test)} test samples with {len(available_columns)} columns")
    
    # Print summary statistics
    mean_abs_error = df_test['absolute_error'].mean()
    mean_pct_error = df_test['percentage_error'].mean()
    print(f"Mean Absolute Error: ${mean_abs_error:.2f}")
    print(f"Mean Percentage Error: {mean_pct_error:.2f}%")
    
    return csv_path

# Training routine

def train():
    # Set seed for reproducibility
    set_seed(SEED)
    
    # Initialize consistent plotting style
    setup_plot_style()
    
    # Create results folder
    results_dir = create_results_folder()
    print(f"Results will be saved to: {results_dir}")
    
    # Resolve CSV path - dataset is in the Code directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(os.path.dirname(script_dir), "final_options_dataset.csv")

    X, y, df_original, vega_weights = load_dataset(csv_path, sample_size=SAMPLE_SIZE)

    # Split data using improved method (time-based or stratified random)
    indices = np.arange(len(X))
    X_train, X_val, X_test, y_train, y_val, y_test, idx_train, idx_val, idx_test = create_data_splits(
        X, y, df_original, indices)
    
    # Verification prints for data splits
    temp_idx = np.concatenate([idx_val, idx_test])
    print("First/last train date:", df_original.iloc[idx_train]['date'].min(),
          df_original.iloc[idx_train]['date'].max())
    print("First/last temp date :", df_original.iloc[temp_idx]['date'].min(),
          df_original.iloc[temp_idx]['date'].max())
    print("Counts -> train/temp/val/test:",
          len(idx_train), len(temp_idx), len(idx_val), len(idx_test))
    
    # Save test identifiers for verification
    test_identifiers = df_original.iloc[idx_test][['date', 'cp_flag', 'strike_price', 'spx_close', 'days_to_maturity', 'mid_price']]
    test_identifiers_path = os.path.join(results_dir, 'test_identifiers.csv')
    test_identifiers.to_csv(test_identifiers_path, index=False)
    
    # Save indices for cross-script verification
    np.save(os.path.join(results_dir, 'test_indices.npy'), idx_test)
    
    temp_identifiers = df_original.iloc[temp_idx][['date', 'cp_flag', 'strike_price', 'spx_close', 'days_to_maturity', 'mid_price']]
    temp_identifiers_path = os.path.join(results_dir, 'temp_identifiers.csv')
    temp_identifiers.to_csv(temp_identifiers_path, index=False)
    
    print("Saved test_indices.npy, temp_identifiers.csv and test_identifiers.csv for verification")
    
    # Compute sample weights AFTER splitting (only on training data to prevent leakage)
    sample_weights_train = None
    if USE_MONEYNESS_SQRT_T_WEIGHTING:
        print(f"\n--- Computing Moneyness × √T Sample Weights (Training Set Only) ---")
        df_train = df_original.iloc[idx_train]
        sample_weights_full = compute_moneyness_sqrt_t_weights(df_train)
        sample_weights_train = sample_weights_full if sample_weights_full is not None else None
    
    # Split vega weights if they exist
    vega_train, vega_val, vega_test = None, None, None
    if vega_weights is not None:
        vega_train = vega_weights[idx_train]
        vega_val = vega_weights[idx_val] 
        vega_test = vega_weights[idx_test]
    
    # Use training-only sample weights (no splitting needed since computed on training set only)
    weights_train = sample_weights_train
    weights_val, weights_test = None, None  # Only training weights are used
    
    # Store underlying prices for denormalization if using normalized targets
    underlying_train, underlying_val, underlying_test = None, None, None
    if USE_NORMALIZED_TARGET:
        underlying_train = df_original.iloc[idx_train]['spx_close'].values
        underlying_val = df_original.iloc[idx_val]['spx_close'].values
        underlying_test = df_original.iloc[idx_test]['spx_close'].values
    
    # Adjust training parameters for normalized targets
    current_epochs = EPOCHS
    current_lr = LEARNING_RATE
    
    if USE_NORMALIZED_TARGET:
        print(f"\n--- Adjusted Training Parameters for Normalized Target ---")
        print(f"Learning rate: {current_lr:.1e} (reduced from {LEARNING_RATE:.1e})")
        print(f"Epochs: {current_epochs}")
    
    # Apply scaling AFTER splitting to prevent data leakage
    # Always scale features (X), but scale targets (y) based on configuration
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    X_train = scaler_x.transform(X_train)
    X_val = scaler_x.transform(X_val)
    X_test = scaler_x.transform(X_test)
    
    # Target scaling logic to prevent double normalization
    scaler_y = None
    if USE_NORMALIZED_TARGET:
        print("Using normalized targets (price/S) - no additional target scaling applied")
        # Targets are already normalized by underlying price, no additional scaling

    elif USE_ZSCORE_SCALING:
        print("Applying Z-Score scaling to targets (mean=0, std=1)")
        scaler_y = StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
        y_train = scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    else:
        print("No target scaling applied - using raw dollar prices")
    
    # Clip and renormalize vega weights (no extra scaling; calculate_black_scholes_vega already handles S-normalization)
    if USE_VEGA_WEIGHTED_LOSS and vega_train is not None:
        print(f"\n--- Clipping and Renormalizing Vega Weights ---")

        # Tight clipping bounds to stabilize training
        VEGA_MIN_CLIP = 0.5
        VEGA_MAX_CLIP = 2.0

        vega_train = np.clip(vega_train, VEGA_MIN_CLIP, VEGA_MAX_CLIP)
        if vega_val is not None:
            vega_val = np.clip(vega_val, VEGA_MIN_CLIP, VEGA_MAX_CLIP)
        if vega_test is not None:
            vega_test = np.clip(vega_test, VEGA_MIN_CLIP, VEGA_MAX_CLIP)

        # Batch renormalization to maintain mean=1 and stable effective learning rate
        vega_train_mean = float(vega_train.mean())
        if vega_train_mean == 0 or not np.isfinite(vega_train_mean):
            vega_train_mean = 1.0
        vega_train = vega_train / vega_train_mean
        if vega_val is not None:
            vega_val = vega_val / vega_train_mean
        if vega_test is not None:
            vega_test = vega_test / vega_train_mean

        # Ensure no NaNs travel into the Dataset
        vega_train = np.nan_to_num(vega_train, nan=1.0, posinf=1.0, neginf=1.0)
        if vega_val is not None:
            vega_val = np.nan_to_num(vega_val, nan=1.0, posinf=1.0, neginf=1.0)
        if vega_test is not None:
            vega_test = np.nan_to_num(vega_test, nan=1.0, posinf=1.0, neginf=1.0)

        print(f"Vega range after clipping: {vega_train.min():.4f} - {vega_train.max():.4f}")
        print(f"Vega mean after renormalization: {vega_train.mean():.4f} (target=1.0)")
        print(f"Clipping bounds: [{VEGA_MIN_CLIP:.4f}, {VEGA_MAX_CLIP:.4f}]")
        print(f"Renormalization factor: {vega_train_mean:.6f}")

    # Create datasets with optional vega weights and sample weights
    train_dataset = OptionPricingDataset(X_train, y_train, vega_train, weights_train)
    val_dataset = OptionPricingDataset(X_val, y_val, vega_val, weights_val)
    test_dataset = OptionPricingDataset(X_test, y_test, vega_test, weights_test)
    
    # Note: Moneyness × √T weighting should be done via per-sample loss weighting instead
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if USE_MONEYNESS_SQRT_T_WEIGHTING and weights_train is not None:
        print("Note: Moneyness × √T weighting enabled via per-sample loss weighting (not WeightedRandomSampler)")
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Using device: {DEVICE}")
    print(f"Architecture: {ARCHITECTURE_NAME} -> {LAYER_SIZES}")
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"Seed: {SEED}")
    
    # Print learning rate warmup configuration
    if USE_LR_WARMUP:
        print(f"Learning rate warmup: {WARMUP_EPOCHS} epochs (start: {current_lr/WARMUP_EPOCHS:.2e} -> target: {current_lr:.2e})")
    else:
        print(f"Learning rate warmup: disabled")

    model = OptionPricingMLP(input_dim=X.shape[1]).to(DEVICE)
    
    # Print model information with correct parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Weight initialization: {'Kaiming (He)' if USE_KAIMING_INIT else 'Xavier (Glorot)'}")
    
    # Configure optimizer based on settings
    if USE_ADAMW:
        # AdamW with no weight decay on bias and normalization parameters
        no_decay = ['bias']  # Parameters that should not have weight decay
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': WEIGHT_DECAY
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=current_lr)
        print(f"Using AdamW optimizer with selective weight decay (lr={current_lr}, wd={WEIGHT_DECAY})")
    else:
        # Standard Adam with weight decay
        optimizer = optim.Adam(model.parameters(), lr=current_lr, weight_decay=WEIGHT_DECAY)
        print(f"Using Adam optimizer with weight decay (lr={current_lr}, wd={WEIGHT_DECAY})")
    
    # Select loss function based on configuration
    if USE_VEGA_WEIGHTED_LOSS and vega_weights is not None:
        criterion = VegaWeightedHuberLoss()  # Will use default delta based on USE_NORMALIZED_TARGET
        print(f"Using Vega-Weighted Huber Loss with vega {'normalized also using strike price' if USE_NORMALIZED_TARGET else 'not normalized also using strike price'}")
    else:
        criterion = nn.HuberLoss(delta=1.0, reduction='none')  # Standard Huber loss, per-sample
        print("Using standard Huber loss")
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, 
                                patience=SCHEDULER_PATIENCE, min_lr=SCHEDULER_MIN_LR)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, 
                                 min_delta=EARLY_STOPPING_MIN_DELTA, 
                                 restore_best_weights=True)
    
    # Initialize model checkpointing
    model_save_path = os.path.join(results_dir, MODEL_SAVE_PATH)
    checkpoint = ModelCheckpoint(filepath=model_save_path, 
                               monitor='val_loss', 
                               save_best_only=True, 
                               mode='min', 
                               save_optimizer=SAVE_OPTIMIZER_STATE)
    
    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(1, current_epochs + 1):
        # Apply learning rate warmup if enabled
        if USE_LR_WARMUP and epoch <= WARMUP_EPOCHS:
            warmup_lr = get_lr_with_warmup(epoch - 1, WARMUP_EPOCHS)  # epoch-1 for 0-indexing
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        elif USE_LR_WARMUP and epoch == WARMUP_EPOCHS + 1:
            # Reset to base learning rate after warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # Apply weight annealing if enabled
        annealing_factor = 1.0
        if USE_WEIGHT_ANNEALING:
            annealing_factor = get_weight_annealing_factor(epoch - 1, WEIGHT_ANNEALING_EPOCHS)  # epoch-1 for 0-indexing
            if hasattr(criterion, 'set_annealing_factor'):
                criterion.set_annealing_factor(annealing_factor)
            # Weight annealing factor applied silently
        
        model.train()
        running_loss = 0.0
        
        for batch_data in train_loader:
            # Always unpack 4-tuple: (features, targets, vega_weights, sample_weights)
            xb, yb, vega_b, sample_b = batch_data
            xb, yb, vega_b, sample_b = xb.to(DEVICE), yb.to(DEVICE), vega_b.to(DEVICE), sample_b.to(DEVICE)
                
            optimizer.zero_grad()
            preds = model(xb)
            
            if USE_VEGA_WEIGHTED_LOSS and vega_b is not None:
                # For vega-weighted loss, the criterion already handles the vega weighting
                # and returns per-sample losses
                
                # Apply sample weights for moneyness×√T weighting if enabled
                if USE_MONEYNESS_SQRT_T_WEIGHTING and sample_b is not None:
                    # Scale by sample weights (moneyness × √T) with annealing
                    loss_vec = criterion(preds, yb, vega_b)  # Returns per-sample losses
                    annealed_sample_weights = apply_sample_weight_annealing(sample_b.view(-1), annealing_factor)
                    loss_vec = loss_vec * annealed_sample_weights
                
                # Take mean over batch
                loss = loss_vec.mean()
            else:
                # Standard loss without vega weighting
                loss_vec = criterion(preds, yb)  # Returns per-sample losses
                
                # Apply sample weights if enabled
                if USE_MONEYNESS_SQRT_T_WEIGHTING and sample_b is not None:
                    annealed_sample_weights = apply_sample_weight_annealing(sample_b.view(-1), annealing_factor)
                    loss_vec = loss_vec * annealed_sample_weights
                
                # Take mean over batch
                loss = loss_vec.mean()
                
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                # Always unpack 4-tuple: (features, targets, vega_weights, sample_weights)
                xb, yb, vega_b, sample_b = batch_data
                xb, yb, vega_b, sample_b = xb.to(DEVICE), yb.to(DEVICE), vega_b.to(DEVICE), sample_b.to(DEVICE)
                    
                preds = model(xb)
                
                if USE_VEGA_WEIGHTED_LOSS and vega_b is not None:
                    # For vega-weighted validation loss
                    loss_vec = criterion(preds, yb, vega_b)  # Returns per-sample losses
                    
                    # Apply sample weights if enabled
                    if USE_MONEYNESS_SQRT_T_WEIGHTING and sample_b is not None:
                        loss_vec = loss_vec * sample_b.view(-1)
                    
                    batch_loss = loss_vec.mean()
                else:
                    # Standard validation loss
                    loss_vec = criterion(preds, yb)  # Returns per-sample losses
                    
                    # Apply sample weights if enabled
                    if USE_MONEYNESS_SQRT_T_WEIGHTING and sample_b is not None:
                        loss_vec = loss_vec * sample_b.view(-1)
                    
                    batch_loss = loss_vec.mean()
                
                val_loss += batch_loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        
        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save checkpoint if this is the best model so far
        checkpoint(val_loss, model, optimizer, epoch=epoch, 
                  train_loss=train_loss, architecture=LAYER_SIZES)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        if epoch == 1 or epoch % 10 == 0 or epoch == current_epochs:
            current_lr_actual = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{current_epochs}  |  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}  |  LR: {current_lr_actual:.2e}")

    # Train and Test evaluation
    model.eval()
    # Collect train predictions/targets
    train_preds, train_trues = [], []
    with torch.no_grad():
        for batch_data in train_loader:
            xb, yb, vega_b, sample_b = batch_data
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy().flatten()
            train_preds.extend(out)
            train_trues.extend(yb.numpy().flatten())

    # Collect test predictions/targets
    preds, trues = [], []
    with torch.no_grad():
        for batch_data in test_loader:
            # Unpack 4-tuple: (features, targets, vega_weights, sample_weights)
            xb, yb, vega_b, sample_b = batch_data
                
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy().flatten()
            preds.extend(out)
            trues.extend(yb.numpy().flatten())

    # Convert predictions and targets back to dollar prices based on scaling configuration
    if USE_NORMALIZED_TARGET:
        print(f"\n--- Converting Normalized Predictions to Dollar Prices (multiply by S) ---")
        print(f"Predictions before conversion: {np.array(preds).min():.4f} - {np.array(preds).max():.4f}")
        print(f"True values before conversion: {np.array(trues).min():.4f} - {np.array(trues).max():.4f}")
        
        # Convert normalized predictions (price/S) back to dollar prices
        preds = np.array(preds) * underlying_test
        trues = np.array(trues) * underlying_test
        # Convert train as well
        train_preds = np.array(train_preds) * underlying_train
        train_trues = np.array(train_trues) * underlying_train
        
        print(f"Predictions after conversion: ${preds.min():.2f} - ${preds.max():.2f}")
        print(f"True values after conversion: ${trues.min():.2f} - ${trues.max():.2f}")

    elif USE_ZSCORE_SCALING and scaler_y is not None:
        print(f"\n--- Inverse Z-Score Transform to Dollar Prices ---")
        # Inverse transform Z-score scaled predictions back to dollar prices
        preds = scaler_y.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        trues = scaler_y.inverse_transform(np.array(trues).reshape(-1, 1)).flatten()
        train_preds = scaler_y.inverse_transform(np.array(train_preds).reshape(-1, 1)).flatten()
        train_trues = scaler_y.inverse_transform(np.array(train_trues).reshape(-1, 1)).flatten()
        print(f"Predictions after inverse transform: ${preds.min():.2f} - ${preds.max():.2f}")
        print(f"True values after inverse transform: ${trues.min():.2f} - ${trues.max():.2f}")
    else:
        print(f"\n--- Using Raw Dollar Predictions (no scaling conversion needed) ---")
        # Predictions are already in dollar prices
        preds = np.array(preds)
        trues = np.array(trues)
        train_preds = np.array(train_preds)
        train_trues = np.array(train_trues)
        print(f"Raw predictions: ${preds.min():.2f} - ${preds.max():.2f}")
        print(f"Raw true values: ${trues.min():.2f} - ${trues.max():.2f}")

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(trues, preds)
    train_metrics = calculate_comprehensive_metrics(train_trues, train_preds)
    # Train/test R^2 and adjusted R^2
    r2_train = r2_score(train_trues, train_preds)
    r2_test = metrics['r2']
    p_features = X_test.shape[1]
    n_train = len(train_trues)
    n_test = len(trues)
    adj_r2_train = 1 - (1 - r2_train) * (n_train - 1) / max(1, (n_train - p_features - 1))
    adj_r2_test = 1 - (1 - r2_test) * (n_test - 1) / max(1, (n_test - p_features - 1))

    # Shapiro-Wilk normality test on residuals (first 20,000 samples)
    residuals_for_test = preds - trues
    n_shapiro = min(5000, len(residuals_for_test))
    try:
        sh_stat, sh_p = stats.shapiro(residuals_for_test[:n_shapiro])
        print(f"Shapiro-Wilk on first {n_shapiro} residuals: W={sh_stat:.6f}, p={sh_p:.6e} (Note: SciPy warns p-values may be inaccurate for N>5000)")
    except Exception as e:
        print(f"Shapiro-Wilk test failed: {e}")
    
    # Export predictions CSV with original unscaled test data
    X_test_original = scaler_x.inverse_transform(X_test)
    export_predictions_csv(df_original, idx_test, X_test_original, trues, preds, results_dir)

    print("\n--- Train Metrics ---")
    print(f"MSE  : {train_metrics['mse']:.4f}")
    print(f"RMSE : {np.sqrt(train_metrics['mse']):.4f}")
    print(f"MAE  : {train_metrics['mae']:.4f}")
    print(f"R² (Train)       : {r2_train:.4f}")
    print(f"Adj R² (Train)   : {adj_r2_train:.4f}")
    
    print("\n--- Test Metrics ---")
    print(f"MSE  : {metrics['mse']:.4f}")
    print(f"RMSE : {np.sqrt(metrics['mse']):.4f}")
    print(f"MAE  : {metrics['mae']:.4f}")
    print(f"R² (Test)        : {r2_test:.4f}")
    print(f"Adj R² (Test)    : {adj_r2_test:.4f}")
    
    # Create df_test from original dataframe using test indices
    df_test = df_original.iloc[idx_test].copy()
    
    # Calculate bucketed metrics (critical for normalized targets)
    moneyness_metrics, time_metrics = calculate_bucketed_metrics(trues, preds, df_test, results_dir)
    
    # Compute bootstrap confidence intervals
    compute_bootstrap_confidence_intervals(trues, preds, results_dir)
    
    # Generate comprehensive visualizations
    print("\n--- Generating Visualizations ---")
    plot_loss_curves(train_losses, val_losses, results_dir)
    plot_true_vs_predicted(trues, preds, results_dir, df_test)
    plot_residuals_histograms(trues, preds, results_dir)
    plot_residuals_histograms(trues, preds, results_dir, use_density=True)
    plot_residuals_vs_features(trues, preds, df_test, results_dir)
    plot_model_performance_summary(train_losses, val_losses, trues, preds, results_dir, metrics)
    
    # Add comprehensive MAE and MAPE analysis
    df_test['predicted_price'] = preds
    create_mae_analysis(df_test, results_dir)
    create_mape_analysis(df_test, results_dir)
    
    # Add comprehensive diagnostic statistics
    write_comprehensive_diagnostics(df_test, results_dir)
    
    # Advanced option pricing plots (controlled by GENERATE_ALL_PLOTS flag)
    if GENERATE_ALL_PLOTS:
        print("\n--- Generating Advanced Option Pricing Plots ---")
        plot_normalized_error_histogram(trues, preds, df_test, results_dir)
        
        # For vega-weighted error plotting, ensure we're using the correct vega units
        if vega_test is not None and USE_VEGA_WEIGHTED_LOSS:
            if USE_NORMALIZED_TARGET:
                # For normalized targets, we need to use the original vega (not scaled by S)
                # since we've already converted predictions back to dollar prices
                S_test = df_original.iloc[idx_test]['spx_close'].values
                vega_plot = vega_test * S_test  # Convert back to dollar vega for plotting
                print(f"Using dollar vega for plotting with normalized targets")
            else:
                # For raw dollar targets, use vega as is
                vega_plot = vega_test
                print(f"Using raw vega for plotting with dollar targets")
            
            plot_vega_weighted_error(trues, preds, vega_plot, results_dir)
        
        plot_error_pivot_grid(trues, preds, df_test, results_dir)
    
    print("\n--- Training Complete ---")
    print(f"Final epoch: {len(train_losses)}")
    print(f"Best validation loss: {min(val_losses):.6f}")
    print(f"Best model saved to: {model_save_path}")
    print(f"All results saved to: {results_dir}")
    
    # Load and verify the best model
    print("\n--- Model Checkpoint Info ---")
    try:
        checkpoint_info = torch.load(model_save_path, map_location='cpu')
        print(f"Checkpoint epoch: {checkpoint_info.get('epoch', 'N/A')}")
        print(f"Checkpoint validation loss: {checkpoint_info.get('score', 'N/A'):.6f}")
        print(f"Model architecture: {checkpoint_info.get('architecture', 'N/A')}")
    except Exception as e:
        print(f"Could not load checkpoint info: {e}")
    
    # Save comprehensive training summary to text file
    summary_path = os.path.join(results_dir, 'training_summary.txt')
    
    # Calculate detailed error statistics using centralized function
    residuals, abs_residuals, pct_error, abs_pct_error = calculate_residuals_and_errors(trues, preds)
    
    # Calculate bid-ask spread accuracy (% of predictions within bid-ask spread)
    df_test = df_original.iloc[idx_test].copy()
    bid_prices = df_test['best_bid'].values
    ask_prices = df_test['best_offer'].values
    
    # Check if predictions are within bid-ask spread
    in_spread = (preds >= bid_prices) & (preds <= ask_prices)
    overall_spread_accuracy = np.mean(in_spread) * 100
    
    # Extract features using consistent indices
    features = extract_features_from_test_data(X_test_original)
    spx_close = features['spx_close']
    strike_price = features['strike_price']
    days_to_maturity = features['days_to_maturity']
    
    # Moneyness and time bucket analysis
    moneyness = spx_close / strike_price
    time_to_expiration = days_to_maturity / DAYS_PER_YEAR  # Convert to years
    
    MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
    MONEYNESS_BIN_LABELS = ['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)']
    TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]
    TIME_BIN_LABELS = ['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)']
    
    moneyness_buckets = pd.cut(moneyness, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)
    time_buckets = pd.cut(time_to_expiration, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MLP NEURAL NETWORK - COMPREHENSIVE TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Architectural Choices Section
        f.write("ARCHITECTURAL CHOICES & DESIGN DECISIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Activation Function: LeakyReLU (slope={LEAKY_RELU_SLOPE})\n")
        f.write(f"Dropout Rate: {DROPOUT_RATE}\n")
        f.write(f"Weight Initialization: {'Kaiming (He)' if USE_KAIMING_INIT else 'Xavier'}\n")
        f.write(f"Optimizer: {'AdamW' if USE_ADAMW else 'Adam'} (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})\n")
        f.write(f"Loss Function: {'Vega-Weighted Huber' if USE_VEGA_WEIGHTED_LOSS else 'Standard Huber'}\n")
        f.write(f"Target Normalization: {'Price/S (normalized)' if USE_NORMALIZED_TARGET else ('Z-score' if USE_ZSCORE_SCALING else 'Raw prices')}\n")
        if USE_MONEYNESS_SQRT_T_WEIGHTING:
            f.write(f"Sample Weighting: Moneyness×√T ({MONEYNESS_BINS}×{SQRT_T_BINS} bins)\n")
            f.write(f"  - Moneyness Range: [{MIN_MONEYNESS:.2f}, {MAX_MONEYNESS:.2f}] in {MONEYNESS_BINS} bins\n")
            f.write(f"  - √T Range: [{MIN_SQRT_T:.2f}, {MAX_SQRT_T:.2f}] in {SQRT_T_BINS} bins\n")
        else:
            f.write("Sample Weighting: None\n")
        f.write(f"Vega Weighting: {'Enabled (ε={EPS})' if USE_VEGA_WEIGHTED_LOSS else 'Disabled'}\n")
        f.write(f"Early Stopping: Patience={EARLY_STOPPING_PATIENCE}, MinΔ={EARLY_STOPPING_MIN_DELTA}\n")
        f.write(f"LR Scheduler: Patience={SCHEDULER_PATIENCE}, Factor={SCHEDULER_FACTOR}\n\n")        

        f.write("-" * 40 + "\n")

        f.write(f"Architecture: {ARCHITECTURE_NAME} -> {LAYER_SIZES}\n")
        
        # Get feature names for current configuration
        _, feature_names = _create_feature_matrix(df_original.iloc[:1])  # Use first row to get feature names
        f.write(f"Input Features ({len(feature_names)}): {', '.join(feature_names)}\n")
        
        f.write(f"Learning Rate Warmup: {'Enabled' if USE_LR_WARMUP else 'Disabled'}{f' ({WARMUP_EPOCHS} epochs)' if USE_LR_WARMUP else ''}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Max Epochs: {EPOCHS}\n")
        f.write(f"Random Seed: {SEED}\n")
        f.write(f"Device: {DEVICE}\n\n")
        
        # Data Filtering Configuration
        f.write("DATA FILTERING CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Zero Volume Inclusion: {ZERO_VOLUME_INCLUSION*100:.0f}%\n")
        f.write(f"Moneyness Bounds: {MIN_MONEYNESS} to {MAX_MONEYNESS}\n")
        f.write(f"Days to Maturity Range: {MIN_DAYS_TO_MATURITY} to {MAX_DAYS_TO_MATURITY}\n")
        f.write(f"Spread Filter: {'Enabled' if SPREAD_FILTER_ENABLED else 'Disabled'}\n")
        if SPREAD_FILTER_ENABLED:
            f.write(f"  - Min Spread (%): {MIN_SPREAD_PCT}\n")
            f.write(f"  - Max Spread (%): {MAX_SPREAD_PCT}\n")
        else:
            f.write(f"  - Legacy Max Spread Ratio (mid): {MAX_SPREAD_RATIO}\n")
        f.write(f"Sample Size Used: {len(trues):,}\n\n")
        
        # Training Results
        f.write("TRAINING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Final Epoch: {len(train_losses)}\n")
        f.write(f"Best Validation Loss: {min(val_losses):.6f}\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.6f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n\n")
        
        # Overall Model Performance
        f.write("OVERALL MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        # Train
        f.write("Train Metrics:\n")
        f.write(f"  - MSE : {train_metrics['mse']:.4f}\n")
        f.write(f"  - RMSE: {np.sqrt(train_metrics['mse']):.4f}\n")
        f.write(f"  - MAE : {train_metrics['mae']:.4f}\n")
        # Test
        f.write("Test Metrics:\n")
        f.write(f"  - MSE : {metrics['mse']:.4f}\n")
        f.write(f"  - RMSE: {np.sqrt(metrics['mse']):.4f}\n")
        f.write(f"  - MAE : {metrics['mae']:.4f}\n")
        f.write(f"R-squared (R²) - Test: {r2_test:.4f}\n")
        f.write(f"Adjusted R-squared - Test: {adj_r2_test:.4f}\n")
        f.write(f"R-squared (R²) - Train: {r2_train:.4f}\n")
        f.write(f"Adjusted R-squared - Train: {adj_r2_train:.4f}\n")
        f.write(f"Bid-Ask Spread Accuracy: {overall_spread_accuracy:.2f}%\n\n")
        
        # Prediction Error Statistics
        f.write("PREDICTION ERROR STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write("Raw Errors (Predicted - Actual):\n")
        f.write(f"  Mean Error: ${np.mean(residuals):.4f}\n")
        f.write(f"  Median Error: ${np.median(residuals):.4f}\n")
        f.write(f"  Std Dev of Errors: ${np.std(residuals):.4f}\n")
        f.write(f"  Min Error: ${np.min(residuals):.4f}\n")
        f.write(f"  Max Error: ${np.max(residuals):.4f}\n\n")
        
        f.write("Absolute Errors:\n")
        f.write(f"  Mean Absolute Error: ${np.mean(abs_residuals):.4f}\n")
        f.write(f"  Median Absolute Error: ${np.median(abs_residuals):.4f}\n")
        f.write(f"  75th Percentile: ${np.percentile(abs_residuals, 75):.4f}\n")
        f.write(f"  90th Percentile: ${np.percentile(abs_residuals, 90):.4f}\n")
        f.write(f"  95th Percentile: ${np.percentile(abs_residuals, 95):.4f}\n")
        f.write(f"  99th Percentile: ${np.percentile(abs_residuals, 99):.4f}\n\n")
        
        f.write("Percentage Errors:\n")
        f.write(f"  Mean Percentage Error: {np.mean(pct_error):.2f}%\n")
        f.write(f"  Mean Absolute Percentage Error: {np.mean(abs_pct_error):.2f}%\n")
        f.write(f"  Median Absolute Percentage Error: {np.median(abs_pct_error):.2f}%\n")
        f.write(f"  75th Percentile Abs %Error: {np.percentile(abs_pct_error, 75):.2f}%\n")
        f.write(f"  90th Percentile Abs %Error: {np.percentile(abs_pct_error, 90):.2f}%\n")
        f.write(f"  95th Percentile Abs %Error: {np.percentile(abs_pct_error, 95):.2f}%\n\n")
        
        # Use the same bucketing approach as calculate_bucketed_metrics (which works correctly)
        # Extract features using the same method as calculate_bucketed_metrics
        df_test = df_original.iloc[idx_test].copy()
        spx_close_correct = df_test['spx_close'].values
        strike_price_correct = df_test['strike_price'].values
        days_to_maturity_correct = df_test['days_to_maturity'].values
        
        # Calculate moneyness and time correctly
        moneyness_correct = spx_close_correct / strike_price_correct
        time_to_expiration_correct = days_to_maturity_correct / DAYS_PER_YEAR
        
        # Create buckets using the same approach as calculate_bucketed_metrics
        MONEYNESS_BINS_EDGES = [0, 0.9, 1.1, np.inf]
        MONEYNESS_BIN_LABELS = ['OTM (<0.9)', 'ATM (0.9-1.1)', 'ITM (>1.1)']
        TIME_BINS_EDGES = [0, 0.083, 0.25, 0.5, 0.75, 1.0, np.inf]
        TIME_BIN_LABELS = ['≤1M (≤30d)', '1-3M (31-91d)', '3-6M (92-182d)', '6-9M (183-274d)', '9-12M (275-365d)', '>12M (>365d)']
        
        moneyness_buckets_correct = pd.cut(moneyness_correct, bins=MONEYNESS_BINS_EDGES, labels=MONEYNESS_BIN_LABELS, include_lowest=True)
        time_buckets_correct = pd.cut(time_to_expiration_correct, bins=TIME_BINS_EDGES, labels=TIME_BIN_LABELS, include_lowest=True)
        
        # Create analysis DataFrame for groupby approach
        analysis_df = pd.DataFrame({
            'abs_residual': abs_residuals,
            'abs_pct_error': abs_pct_error,
            'residual': residuals,
            'in_spread': in_spread,
            'moneyness_bucket': moneyness_buckets_correct,
            'time_bucket': time_buckets_correct
        })
        
        # Moneyness Bucket Analysis
        f.write("MONEYNESS BUCKET ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Performance by Moneyness (S/K) Categories:\n\n")
        
        # Calculate metrics by moneyness buckets using groupby (copy from calculate_bucketed_metrics)
        moneyness_analysis = analysis_df.groupby('moneyness_bucket').agg({
            'abs_residual': 'mean',
            'abs_pct_error': 'mean',
            'residual': 'count'
        }).reset_index()
        moneyness_analysis.columns = ['bucket', 'mae', 'mape', 'count']
        
        for _, row in moneyness_analysis.iterrows():
            bucket = row['bucket']
            if pd.isna(bucket):
                continue
            f.write(f"{bucket:>10} (n={row['count']:,}):\n")
            f.write(f"  Mean Absolute Error: ${row['mae']:.4f}\n")
            f.write(f"  Mean Abs % Error: {row['mape']:.2f}%\n\n")
        
        # Time to Expiration Analysis
        f.write("TIME TO EXPIRATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Performance by Time to Expiration Categories:\n\n")
        
        # Calculate metrics by time buckets using groupby (copy from calculate_bucketed_metrics)
        time_analysis = analysis_df.groupby('time_bucket').agg({
            'abs_residual': 'mean',
            'abs_pct_error': 'mean',
            'residual': 'count'
        }).reset_index()
        time_analysis.columns = ['bucket', 'mae', 'mape', 'count']
        
        for _, row in time_analysis.iterrows():
            bucket = row['bucket']
            if pd.isna(bucket):
                continue
            f.write(f"{bucket:>6} (n={row['count']:,}):\n")
            f.write(f"  Mean Absolute Error: ${row['mae']:.4f}\n")
            f.write(f"  Mean Abs % Error: {row['mape']:.2f}%\n\n")
        
        # Model Diagnostics
        f.write("MODEL DIAGNOSTICS\n")
        f.write("-" * 40 + "\n")
        f.write("Residuals Normality Tests:\n")
        f.write(f"  Residuals Mean: {np.mean(residuals):.6f}\n")
        f.write(f"  Residuals Std Dev: {np.std(residuals):.6f}\n")
        f.write(f"  Skewness: {stats.skew(residuals):.4f}\n")
        f.write(f"  Kurtosis: {stats.kurtosis(residuals):.4f}\n")
        
        # Shapiro-Wilk test on first 20,000 residuals (SciPy note: p-values may be inaccurate for N>5000)
        n_shapiro = min(20000, len(residuals))
        try:
            shapiro_stat, shapiro_p = stats.shapiro(residuals[:n_shapiro])
            f.write(f"  Shapiro-Wilk (first {n_shapiro}): W={shapiro_stat:.6f}, p-value={shapiro_p:.6e}\n")
        except Exception as e:
            f.write(f"  Shapiro-Wilk test failed: {e}\n")
        
        f.write(f"\nModel Complexity:\n")
        # Calculate total parameters properly: weights + biases for all layers
        if len(LAYER_SIZES) > 0:
            input_size = _get_input_size_for_feature_mode()
            total_params = input_size * LAYER_SIZES[0] + LAYER_SIZES[0]  # Input to first hidden
            for i in range(len(LAYER_SIZES)-1):
                total_params += LAYER_SIZES[i] * LAYER_SIZES[i+1] + LAYER_SIZES[i+1]  # Hidden to hidden
            total_params += LAYER_SIZES[-1] * 1 + 1  # Last hidden to output
        else:
            input_size = _get_input_size_for_feature_mode()
            total_params = input_size * 1 + 1  # Direct input to output
        
        f.write(f"  Total Parameters (approx): {total_params:,}\n")
        f.write(f"  Number of Hidden Layers: {len(LAYER_SIZES)}\n")
        f.write(f"  Layer Sizes: {LAYER_SIZES}\n\n")
        
        # Files Generated
        f.write("FILES GENERATED\n")
        f.write("-" * 40 + "\n")
        f.write("Visualization Files:\n")
        f.write("- loss_curves.png (Training progress)\n")
        f.write("- prediction_analysis.png (True vs predicted analysis)\n")
        f.write("- mlp_diagnostics_part1.png (Core diagnostic plots: residuals vs fitted, distribution, Q-Q, scale-location)\n")
        f.write("- mlp_diagnostics_part2.png (Feature-based diagnostics: residuals vs maturity/moneyness, error distributions)\n")
        f.write("- mlp_error_analysis_part1.png (Price-based error analysis: error vs price, quantiles, volatility)\n")
        f.write("- mlp_error_analysis_part2.png (Advanced error analysis: moneyness/time buckets, cumulative distribution, statistics)\n")
        f.write("- model_performance_summary.png (Comprehensive model performance dashboard)\n\n")
        f.write("Data Files:\n")
        f.write("- mlp_predictions.csv (Detailed predictions with inputs and errors)\n")
        f.write(f"- {MODEL_SAVE_PATH} (Best model checkpoint)\n")
        f.write("- training_summary.txt (This summary file)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF MLP TRAINING SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"Training summary saved to: {summary_path}")

if __name__ == "__main__":
    train()
