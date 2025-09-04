Overview
Goal: End-to-end SPX European call option pricing analysis using regression and MLPs, with Black–Scholes as reference.

Project Structure
- 00Data/ — auxiliary macroeconomic series.
- 01General/ — data exploration and processing utilities and figures.
- 02Regression/ — baseline regression modeling.
- 03Black_Scholes/ — reference analysis using Black–Scholes (vectorized implementation).
- 04MLP1/ — first neural network approach (baseline MLP).
- 05MLP2/ — improved NN with config-driven training, weighting, and plotting.
- 06Other_Scrap_Codes/ — experimental and archived code (MLP3, Residual method, diagrams, etc.).
- final_options_dataset.csv — main dataset (root).

Environment Setup
- Python 3.12 recommended (virtualenv or venv).
- Create and activate virtual environment (macOS/Linux):
  python3 -m venv .venv
  source .venv/bin/activate
- Install dependencies:
  pip install numpy pandas scikit-learn scipy matplotlib seaborn pyyaml torch torchvision torchaudio
- Optional: Freeze dependencies
  pip freeze > requirements.txt

Data
- Main dataset: final_options_dataset.csv (project root).
- Data sources and additional macro series: 00Data/
- Important correction: strike prices in raw options are per-contract (100x). Processing divides strikes by 100 in 01General/data_processing.py to get per-share values used by models and Black–Scholes.
- Coverage: ~4.73M options (2016-01-04 to 2019-09-12) with added features: open_interest, epu_index, equity_uncertainty, equity_volatility.

Key Components
- 01General/
  - data_explorer.py — inspect distributions, nulls, ranges, relationships.
  - data_processing.py — preprocessing/feature augmentation (already baked into final_options_dataset.csv; running is optional).
  - eda_all_figures/ — pre-saved visual summaries.

- 02Regression/
  - Regression.py — baseline models, diagnostics.
  - results/ — metrics, plots.

- 03Black_Scholes/
  - black_scholes_analysis.py — vectorized BS pricing and Greeks, error breakdowns.
  - results/ — analysis figures and metrics.

- 04MLP1/
  - MLP1.py — baseline neural network for pricing.
    - Normalized target option (price/S) and automatic training parameter adjustment when enabled.
    - Vega-weighted loss with conservative clipping and moneyness×√T bin weighting.
    - Configurable feature modes including core_only (strike_price, dividend_rate, risk_free_rate, days_to_maturity, spx_close) and default feature set.
    - Configurable bid–ask spread filtering by percentage (applied early in load_dataset).
    - Time-based splitting (by date fraction or cutoff date) in addition to random splits; stratification fix when using normalized targets.
    - Detailed bucketed metrics (by moneyness and time-to-maturity), saved to CSV and shown during training.
  - vol_graphs.py — IV smiles/term structures and related graphs.
  - results/ — loss curves, error plots, bucketed metrics, best_model.pth, summaries.

- 05MLP2/
  - MLP2.py — improved training with simplified feature inputs
  - model_comparison.py — compare different pricing models.
  - results/ — training_summary.txt, best_model.pth, bucketed_metrics.csv, predictions, plots, and run folders.

Quick Start
1) Ensure environment is ready and final_options_dataset.csv exists at repo root.
2) Run baselines:
  python 02Regression/Regression.py
  python 03Black_Scholes/black_scholes_analysis.py
3) Train MLP1:
  python 04MLP1/MLP1.py
4) Train MLP2 (recommended):
  python 05MLP2/MLP2.py

MLP Configuration Highlights (05MLP2/MLP2.py)
- Target normalization: USE_NORMALIZED_TARGET=True enables price/S training with tuned LR/epochs.
- Loss weighting: USE_VEGA_WEIGHTED_LOSS=True emphasizes ATM; conservative vega clipping and balanced binning (e.g., 8×8).
- Data quality filters:
  - Bid–ask spread percentage filter: SPREAD_FILTER_ENABLED with MIN_SPREAD_PCT/MAX_SPREAD_PCT.
  - Volume and other filters may be applied upstream (see dataset and MLP1 notes).
- Feature selection: FEATURE_MODE supports core_only and default_only, among others.
- Splits:
  - USE_TIME_BASED_SPLIT and USE_TIME_PERCENT_SPLIT to split by time fraction or cutoff date (TIME_SPLIT_FRACTION/TIME_SPLIT_DATE).
  - Stratification fix ensures fair validation when using normalized targets.
- Best-known stable settings example: normalized target on, no z-score scaling, vega-weighted loss, AdamW, low dropout, gentle LR scheduling, early stopping around ~50 epochs.

Reproducibility and Splits
- Seeds are configurable; runs are deterministic where possible.
- Note: Different scripts may reset indices or filter prior to splitting, which can change row order and lead to different splits even with the same seed. Black–Scholes resets after filtering; MLP2 filters then subsets before splitting.

Tips
- Most artifacts are pre-generated in each results/ directory; you can review without re-running.
- If re-running, verify packages and the dataset path. For quick diagnostics, start with Regression, compare with Black–Scholes, then evaluate MLP1/MLP2 results.
