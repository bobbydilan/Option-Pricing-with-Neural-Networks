"""
Package Import Script
Imports all packages listed in requirements.txt for the option pricing MLP project 
These are also imported individually in the scripts, however it is suggested to import them here previously to check for installation issues
"""

# Data manipulation and analysis
import pandas as pd
import numpy as np
import matplotlib

# Machine learning
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep learning with PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision
    print("PyTorch imported successfully")
except ImportError:
    print("PyTorch not available - will use scikit-learn MLPRegressor instead")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Financial data
try:
    import yfinance as yf
    print("yfinance imported successfully")
except ImportError:
    print("yfinance not available")

# Scientific computing
import scipy
from scipy import stats

# Jupyter notebook support
try:
    import jupyter
    print("Jupyter imported successfully")
except ImportError:
    print("Jupyter not available")

# Utilities
import warnings
import os
import sys
from datetime import datetime, timedelta

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
try:
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
except:
    pass

print("\n" + "="*50)
print("PACKAGE IMPORT STATUS")
print("="*50)
print(f"pandas {pd.__version__}")
print(f"numpy {np.__version__}")
print(f"scikit-learn {sklearn.__version__}")
print(f"matplotlib {matplotlib.__version__}")
print(f"seaborn {sns.__version__}")
print(f"scipy {scipy.__version__}")

try:
    print(f"torch {torch.__version__}")
    print(f"torchvision {torchvision.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available - using CPU")
except:
    print("PyTorch not available")

print("="*50)
print("All core packages imported successfully!")
print("Ready for option pricing MLP development with PyTorch.")
print("="*50)
