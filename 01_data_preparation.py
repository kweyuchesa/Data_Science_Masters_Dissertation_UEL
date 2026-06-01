# =====================================================
# ML-AUGMENTED GRAVITY MODEL FOR TRADE FLOWS
# NORTHERN CORRIDOR TRADE FLOW PREDICTION
 # An ML-Augmented Gravity Model with SHAP Interpretability
 # Chesa Kweyu - Masters in Data Science
    # Dissertation Title: Prediction of Trade Flows and Non-Tariff Barrier Impact Analysis Along the Northern 
                          # Corridor Using Augmented Gravity Model and Machine Learning: 
                          # Evidence from Kenya, Uganda, and Rwanda (2017–2024)

# Period: 2017Q1 - 2024Q4 | Countries: Kenya, Uganda, Rwanda

# ========================= Importing Libraries and dependacies ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap
import statsmodels.api as sm

# For PPML (Poisson Pseudo-Maximum Likelihood) - install if needed: pip install pygravity or use statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import poisson

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ========================== 1. DATA LOADING & CLEANING ==========================
print(" - Loading and Cleaning Data - ")

df = pd.read_csv('/home/workdir/attachments/thesis_tradeflowdata.csv')

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# Create time index
df['year'] = df['yr'].astype(int)
df['quarter'] = df['quarter'].str.replace('Q', '').astype(int)
df['time'] = pd.to_datetime(df['year'].astype(str) + '-Q' + df['quarter'].astype(str))
df = df.sort_values('time')

print(f"Dataset shape: {df.shape}")
print(f"Time range: {df['time'].min()} to {df['time'].max()}")
print(f"Unique pairs: {df.groupby(['countryi', 'countryj']).size().count()}")

# Handle missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Fill small missing NTB/logistics with 0 (common in such data)
ntb_cols = [col for col in df.columns if 'ntb' in col]
logistics_cols = ['tr_mbs', 'cdt_mbs', 'wbcnon_nc']
df[ntb_cols + logistics_cols] = df[ntb_cols + logistics_cols].fillna(0)

# Log transform trade flows (add 1 to handle zeros)
df['exp_ij_log'] = np.log(df['exp_ij'] + 1)

print("✅ Data loaded and cleaned.")
