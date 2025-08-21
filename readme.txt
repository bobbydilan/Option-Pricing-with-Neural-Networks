Overview
    Goal: End-to-end SPX European call option pricing analysis using MLPs.

Structure:
    00Data/ — auxiliary macroeconomic series.
    01General/ — data exploration and processing utilities and figures.
    02Regression/ — baseline regression modeling.
    03Black-Scholes/ — reference analysis using Black–Scholes.
    04MLP1/ — first neural network approach.
    05MLP2/ — improved NN with config-driven training and plotting.
    06Other Models & Scrap/ — experimental and archived code.
    final_options_dataset.csv — main dataset.

Environment Setup
    Python 3.12 recommended (virtualenv or venv).
    Create and activate virtual environment:
        macOS/Linux: 
            python3 -m venv .venv
            source .venv/bin/activate
    Install dependencies (if you don’t have a requirements file, install these):
        pip install numpy pandas scikit-learn scipy matplotlib seaborn pyyaml torch torchvision torchaudio
    Optional: Freeze dependencies
        pip freeze > requirements.txt

Data
    Main created dataset: final_options_dataset.csv (project root).
    Data Sources and additional macro series in 00Data/.

Suggested Workflow
    1. General — Setup, Exploration, Processing
        Import and verify environment and packages (see “Environment Setup” above).

        Data Exploration:
            File: 01General/data_explorer.py
            Purpose: Inspect distributions, nulls, ranges, and basic relationships.
            Results: Visual summaries saved under 01General/eda_all_figures/ (already populated).

        Data Processing:
            File: 01General/data_processing.py
            Purpose: Any preprocessing/feature augmentation pipeline.
            Note: Core processed data is already baked into final_options_dataset.csv; running is optional.

        EDA All Figures (Pre-saved)
            Directory: 01General/eda_all_figures/
            Contains: Key EDA plots across moneyness, time-to-maturity, volatility, errors, and more.
            Action: Review figures directly; no need to re-run the scripts unless you want to regenerate.

    2. Regression Baselines
        File: 02Regression/Regression.py
        Purpose: Baseline models and diagnostic plots.
        Outputs (pre-saved): 02Regression/results/ e.g., correlation heatmaps, error-by-feature, metrics txt.

    3. Black–Scholes Analysis
        File: 03Black-Scholes/black_scholes_analysis.py
        Purpose: BS pricing reference, error breakdowns vs empirical data.
        Outputs (pre-saved): 03Black-Scholes/analysis_results/ PNGs and metrics.

    4. MLP1
        File: 04MLP1/MLP1.py
        Purpose: First neural network baseline for pricing.
        Outputs (pre-saved): 04MLP1/results/ includes loss curves, error plots, bucketed metrics, best_model.pth.

    5. MLP2
        Files:
            05MLP2/MLP2_Train Model.py — legacy training script.
            05MLP2/vol_graphs.py — plotting IV smiles/term structures or related graphs.
        Purpose: Improved training with vega-weighted loss, moneyness×√T weighting, normalization options, and richer plots.
        
        Outputs (pre-saved):
            05MLP2/results/
            training_summary.txt
            best_model.pth
            bucketed_metrics.csv
            predictions CSV(s)
            training and error plots

    6. Other Models & Scrap
    Directory: 06Other Models & Scrap/
    Purpose: Experimental or archived approaches (e.g., MLP3/, Residual Method Extra/, Vol/, and script drafts).
    Use: Optional; good for reference, ablations, or further ideas.

Reproducibility
    Seed is configurable. Runs are deterministic when possible.

Device selection:
    general.device: 'auto' uses CUDA if available, otherwise CPU.

Tips
    You can review all results without running code: most artifacts are already in their respective results/ folders.
    If re-running models, ensure final_options_dataset.csv is present and your environment has the required packages.
    For quick diagnostics, start with Regression/outputs, then compare with Black–Scholes, and finally review MLP results.