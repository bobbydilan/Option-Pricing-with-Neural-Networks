SPX European Call Option Pricing — Regression, Black–Scholes & MLPs
====================================================================

Overview
--------
This repository contains a complete end-to-end pipeline for analysing and pricing S&P 500 European call options. 
The project combines traditional regression methods, a vectorised Black–Scholes-Merton reference implementation, 
and modern neural networks (two successive MLP architectures). The work is directly tied to the accompanying 
thesis: “Enhancing Black–Scholes-Merton Option Pricing Accuracy with Neural Networks: A Data-Driven Approach on 
European S&P 500 Index Options.”

The repository is structured so that anyone can reproduce the entire workflow with one key requirement: the 
main options dataset (`final_options_dataset.csv`) must be placed in the repository root. This file is not 
included in the repository due to licensing and size constraints, but it can be requested or re-created from 
OptionMetrics data. All other code, utilities, and result artefacts are included and structured to allow full 
replication. The final_options_dataset.csv can be created directly by the data_processing.py script in 00General 
if the appropriate datasets are provided.

It is important to note that very large intermediate outputs—such as full prediction datasets from trained models—are 
not included here to avoid bloating the repository. This means that certain scripts, like the model comparison tool, 
cannot be executed out of the box because they require those prediction datasets. However, the results of those scripts 
are visible in the respective `results/` folders, and the full process to regenerate them is completely transparent 
from the code.

---

Project Structure
-----------------
The repository follows a logical progression from raw data to baselines, to reference models, and finally to 
advanced neural architectures. Keeping the directory structure consistent is crucial—many scripts assume these 
paths. To run the models below, it is suggested to download these codes in the same folder structure as present 
in this repository.

- 00Data/  
  Contains auxiliary macroeconomic and market time series used as inputs. This includes SPX price data, 
  historical volatility measures, economic policy uncertainty indices, and equity market uncertainty indices. 
  These are not option-specific but provide the macro-financial backdrop. All files here are included. The final
  options panel is represented in all other codes by the single file `final_options_dataset.csv` at the 
  repository root, which must be supplied separately or recreated using above mentioned data. The option-specific
  dataset is not uploaded. The timespan of the uploaded observations is 01.01.2012-01.01.2020.

- 01General/  
  General exploration and preprocessing tools. Scripts allow quick inspection of distributions, missing values, 
  and relationships between variables. Processing scripts harmonise units (e.g., converting strike prices from 
  per-contract to per-share values), create engineered features (forward log-moneyness, sqrt time-to-maturity), 
  and export the clean dataset. Pre-generated exploratory figures are included for reference.

- 02Regression/  
  Implements baseline regressions such as OLS and regularised variants. While simple, these models provide a 
  benchmark against which more sophisticated approaches are compared. The corresponding results—metrics and plots—
  are run, saved and uploaded in the zip file available. If `final_options_dataset.csv` is present, these can be 
  rerun easily.

- 03Black_Scholes/  
  Provides a vectorised Black–Scholes reference implementation. This script computes option prices and Greeks, 
  using historical volatility as input, and then produces error breakdowns by moneyness and time-to-maturity. 
  The results are saved in zip file available, and serve as the classical baseline against which machine 
  learning models are evaluated.

- 04MLP1/  
  First neural network approach. This script is actually the same as the one in MLP2. The difference is
  in the toggle setting, which decides which feature inputs to use. This baseline MLP introduces advanced 
  techniques such as:
  - Normalised target training (price/S).
  - Vega-weighted losses with clipping.
  - Sample reweighting by moneyness × sqrt time.
  - Flexible feature modes and spread-based data filters.
  It includes a `run_experiments.py` script for hyperparameter exploration and a `vol_graphs.py` utility for 
  producing implied volatility style visualisations (smiles, term structures, vega-weighted error maps). 
  The zip files contain multiple run subfolders (run_MLP1.1, run_MLP1.2, etc.) with training logs, 
  plots, metrics, and saved models. Also analysis on implied volatility may be found in the other zip file.

- 05MLP2/  
  Focusing on reduced but engineered feature sets (forward log-moneyness, sqrt T). MLP2 refines the 
  training loop, improves generalisation, and consistently outperforms earlier baselines in critical 
  pricing regions. Scripts include:
  - `MLP2.py`: main training loop with configuration flags.
  - `vol_graphs.py`: regenerates implied-volatility-style visualizations. It now writes outputs into a new 
    `vol_results/` folder (volatility results zip file), organised by run.
  - `model_comparison_diagnostic.py`: compares results across models. Note: full comparison cannot be re-run without 
    the large prediction files, but the code shows the exact process.
  - Results are pre-saved under `results/` (run_MLP2.1, run_MLP2.2, run_MLP2.3), can be re-run using the scripts but
    are also available in the results zip file.

- 06Other_Scrap_Codes/  
  Contains experimental and archived material: early MLP3 experiments, alternative residual approaches, and 
  illustrative diagrams. These are not central to the final analysis but are included for completeness. The code behind 
  the various diagrams present in the thesis is available.

- final_options_dataset.csv  
  The essential dataset for replication. Not included in the repository but required. All analyses assume this file 
  exists in the project root.

---

Environment
-----------
- Python 3.12 recommended
- Create a virtual environment: python3 -m venv .venv
                                source .venv/bin/activate
- Install dependencies: pip install numpy pandas scikit-learn scipy matplotlib seaborn pyyaml torch torchvision torchaudio
- Optional: freeze environment with `pip freeze > requirements.txt`

---

Data Notes
----------
- Coverage: ~4.73 million option records, from 2016-01-04 to 2019-09-12.
- Sources: OptionMetrics (for option data) plus auxiliary series in 00Data.
- Adjustments: Strikes are converted from per-contract to per-share values.
- Engineered features: forward log-moneyness, sqrt time-to-maturity, days_to_maturity, moneyness.

---

Workflow & Results
------------------
The general process is:

1. Exploration & Processing  
 Use 01General scripts to understand and preprocess the dataset.

2. Baselines  
 Run 02Regression for simple models and 03Black_Scholes for the classical reference. Pre-generated results are available.

3. MLP1  
 Train baseline neural nets with MLP1.py, experiment with hyperparameters, and visualise with vol_graphs.py. Results are logged in dedicated run folders.

4. MLP2  
 Use the improved setup (MLP2.py). Each run produces a rich set of diagnostics saved under results/run_MLP2. The vol_graphs.py script creates volatility and implied-volatility style diagnostics under vol_results/.

---

Outputs
-------
Each MLP run produces:
- Training summary and best model weights
- Prediction CSVs (including all feature columns)
- Detailed bucketed metrics
- Error pivot tables
- Multiple diagnostic plots (loss curves, prediction scatterplots, residual histograms, error heatmaps, 
vega-weighted error analyses, implied volatility visuals)

Due to their size, the full prediction CSVs are not included in the GitHub repository. 
To directly see prediction  values, you must re-run the scripts. This means, for example, 
that while the process of model comparison is fully documented in code, you cannot execute it 
without regenerating predictions. However, the already-produced plots and summaries are included 
so that the final insights are still accessible.

---

Reproducibility
---------------
- Seeds are set where possible. Some differences in splits may arise due to filtering and indexing choices.
- All results in `results/` are saved exactly as produced during training and evaluation.
- `vol_graphs.py` has been adjusted so that every run’s volatility analysis is placed in `vol_results/`.

---

Closing
-------
This repository is intended as both a reference and a reproducible workflow for applying machine learning 
to option pricing. It connects theory (Black–Scholes) with practical empirical modelling (regressions and 
neural networks), and emphasises diagnostics and interpretability as much as raw performance.

License: MIT

