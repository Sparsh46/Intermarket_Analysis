import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. DATA COLLECTION
# ============================================================
tickers = {
    "NIFTY_IT":     "^CNXIT",
    "NIFTY_BANK":   "^NSEBANK",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "NIFTY_50":     "^NSEI",
    "IT_US":        "XLK",
    "BANK_US":      "XLF",
    "ENERGY_US":    "XLE",
    "PHARMA_US":    "XLV",
    "SP500":        "^GSPC"
}

raw = yf.download(list(tickers.values()), start="2018-01-01", end="2024-12-31")["Close"]
raw.columns = list(tickers.keys())
monthly = raw.resample('ME').last().pct_change().dropna()
print(f"Data shape: {monthly.shape}")
print(monthly.head())

# ============================================================
# 2. RISK FREE RATE (India: 91-day T-bill proxy = 6% annual)
# ============================================================
RF_annual   = 0.06
RF_monthly  = RF_annual / 12
monthly['RF'] = RF_monthly

# ============================================================
# 3. EXCESS RETURNS
# ============================================================
sectors_india = ["NIFTY_IT", "NIFTY_BANK", "NIFTY_ENERGY", "NIFTY_PHARMA"]
sectors_us    = ["IT_US", "BANK_US", "ENERGY_US", "PHARMA_US"]

for col in sectors_india + sectors_us:
    monthly[f"{col}_excess"] = monthly[col] - monthly['RF']

monthly['MKT_RF_INDIA'] = monthly['NIFTY_50'] - monthly['RF']
monthly['MKT_RF_US']    = monthly['SP500']    - monthly['RF']

# ============================================================
# 4. CONSTRUCT INDIAN FF3 FACTORS (Simplified proxy)
# Using NIFTY_50 as market, size/value proxied from index spreads
# NOTE: For full FF3, use IIM Ahmedabad factors (Agarwalla et al.)
# ============================================================
# Proxy SMB = Small cap index - Large cap index
# Since we don't have small cap here, we use Nifty50 vs sector spread
# This is a simplified version — replace with actual SMB/HML if available

# For now, use US FF3 from Ken French (for US sectors)
# and market factor only for Indian sectors (most common in Indian studies)

print("\nNote: For proper Indian FF3, download factors from:")
print("https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/")

# ============================================================
# 5. FF3 REGRESSION — INDIAN SECTORS (Market factor only proxy)
# ============================================================
print("\n" + "="*60)
print("FF3 REGRESSION RESULTS — INDIAN SECTORS")
print("="*60)

india_betas = {}

for sector in sectors_india:
    y = monthly[f"{sector}_excess"]
    X = sm.add_constant(monthly['MKT_RF_INDIA'])
    model = sm.OLS(y, X).fit()
    india_betas[sector] = model.params['MKT_RF_INDIA']
    print(f"\n{sector}")
    print(f"  Alpha : {model.params['const']:.4f}  (p={model.pvalues['const']:.3f})")
    print(f"  Beta  : {model.params['MKT_RF_INDIA']:.4f}  (p={model.pvalues['MKT_RF_INDIA']:.3f})")
    print(f"  R²    : {model.rsquared:.4f}")

# ============================================================
# 6. FF3 REGRESSION — US SECTORS
# ============================================================
print("\n" + "="*60)
print("FF3 REGRESSION RESULTS — US SECTORS")
print("="*60)

us_betas = {}

for sector in sectors_us:
    y = monthly[f"{sector}_excess"]
    X = sm.add_constant(monthly['MKT_RF_US'])
    model = sm.OLS(y, X).fit()
    us_betas[sector] = model.params['MKT_RF_US']
    print(f"\n{sector}")
    print(f"  Alpha : {model.params['const']:.4f}  (p={model.pvalues['const']:.3f})")
    print(f"  Beta  : {model.params['MKT_RF_US']:.4f}  (p={model.pvalues['MKT_RF_US']:.3f})")
    print(f"  R²    : {model.rsquared:.4f}")

# ============================================================
# 7. ROLLING REGRESSION — 12 MONTH WINDOW
# ============================================================
print("\n" + "="*60)
print("ROLLING BETAS — 12 MONTH WINDOW")
print("="*60)

window = 12

rolling_betas_india = pd.DataFrame(index=monthly.index)
rolling_betas_us    = pd.DataFrame(index=monthly.index)

for sector in sectors_india:
    betas = []
    for end in range(window, len(monthly)+1):
        chunk = monthly.iloc[end-window:end]
        y = chunk[f"{sector}_excess"]
        X = sm.add_constant(chunk['MKT_RF_INDIA'])
        try:
            b = sm.OLS(y, X).fit().params['MKT_RF_INDIA']
        except:
            b = np.nan
        betas.append(b)
    rolling_betas_india[sector] = [np.nan]*(window-1) + betas

for sector in sectors_us:
    betas = []
    for end in range(window, len(monthly)+1):
        chunk = monthly.iloc[end-window:end]
        y = chunk[f"{sector}_excess"]
        X = sm.add_constant(chunk['MKT_RF_US'])
        try:
            b = sm.OLS(y, X).fit().params['MKT_RF_US']
        except:
            b = np.nan
        betas.append(b)
    rolling_betas_us[sector] = [np.nan]*(window-1) + betas

rolling_betas_india.index = monthly.index
rolling_betas_us.index    = monthly.index

# ============================================================
# 8. SHOCK EVENTS
# ============================================================
shocks = {
    "COVID Crash":    ("2020-03", "red"),
    "Fed Hike":       ("2022-03", "orange"),
    "Russia-Ukraine": ("2022-02", "green"),
    "US Banking":     ("2023-03", "purple"),
    "India Election": ("2024-05", "gray"),
}

# ============================================================
# 9. PLOT 1 — MONTHLY RETURNS
# ============================================================
pairs = [
    ("NIFTY_IT",     "IT_US",     "IT Sector"),
    ("NIFTY_BANK",   "BANK_US",   "Banking Sector"),
    ("NIFTY_ENERGY", "ENERGY_US", "Energy Sector"),
    ("NIFTY_PHARMA", "PHARMA_US", "Pharma Sector"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
axes = axes.flatten()

for i, (india_col, us_col, title) in enumerate(pairs):
    ax = axes[i]
    ax.plot(monthly.index, monthly[india_col], label='India', color='#1f77b4', linewidth=1.5)
    ax.plot(monthly.index, monthly[us_col],    label='US',    color='#ff7f0e', linewidth=1.5, linestyle='--')
    for shock_name, (shock_date, shock_color) in shocks.items():
        ax.axvline(pd.to_datetime(shock_date), color=shock_color, linestyle=':', alpha=0.8, linewidth=1.2)
        if i == 0:
            ax.axvline(pd.to_datetime(shock_date), color=shock_color, linestyle=':', alpha=0.8, linewidth=1.2, label=shock_name)
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel("Monthly Return", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8)

fig.suptitle("India vs US Sector Monthly Returns (2018–2024)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("01_monthly_returns.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 10. PLOT 2 — ROLLING BETAS
# ============================================================
sector_pairs = [
    ("NIFTY_IT",     "IT_US",     "IT"),
    ("NIFTY_BANK",   "BANK_US",   "Banking"),
    ("NIFTY_ENERGY", "ENERGY_US", "Energy"),
    ("NIFTY_PHARMA", "PHARMA_US", "Pharma"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
axes = axes.flatten()

for i, (india_col, us_col, title) in enumerate(sector_pairs):
    ax = axes[i]
    ax.plot(rolling_betas_india.index, rolling_betas_india[india_col], label='India Beta', color='#1f77b4', linewidth=2)
    ax.plot(rolling_betas_us.index,    rolling_betas_us[us_col],       label='US Beta',    color='#ff7f0e', linewidth=2, linestyle='--')
    for shock_name, (shock_date, shock_color) in shocks.items():
        ax.axvline(pd.to_datetime(shock_date), color=shock_color, linestyle=':', alpha=0.8, linewidth=1.2)
    ax.axhline(1, color='black', linewidth=0.6, linestyle='--', alpha=0.3)
    ax.set_title(f"{title} — Rolling 12M Beta", fontsize=13, fontweight='bold')
    ax.set_ylabel("Beta", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8)

fig.suptitle("Rolling 12-Month Market Betas — India vs US (2018–2024)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("02_rolling_betas.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# GRANGER CAUSALITY — DAILY (correct frequency)
# ============================================================
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

# Use daily returns instead of monthly
daily = raw.pct_change().dropna()

# Add market excess returns
daily['MKT_RF_INDIA'] = daily['NIFTY_50'] - (0.06/252)  # daily RF
daily['MKT_RF_US']    = daily['SP500']    - (0.06/252)

scaler = StandardScaler()

granger_pairs = [
    ("NIFTY_IT",     "IT_US",     "IT"),
    ("NIFTY_BANK",   "BANK_US",   "Banking"),
    ("NIFTY_ENERGY", "ENERGY_US", "Energy"),
    ("NIFTY_PHARMA", "PHARMA_US", "Pharma"),
]

granger_results = {}

print("\n" + "="*60)
print("GRANGER CAUSALITY — DAILY RETURNS (max lag = 5 days)")
print("="*60)

for india_col, us_col, name in granger_pairs:
    df_gc = daily[[india_col, us_col]].dropna()

    # Scale
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_gc),
        columns=df_gc.columns,
        index=df_gc.index
    )

    granger_results[name] = {}

    print(f"\n{name}")

    # US → India
    print(f"  US → India?")
    res = grangercausalitytests(df_scaled[[india_col, us_col]], maxlag=5, verbose=False)
    for lag in [1, 2, 3, 4, 5]:
        pval = res[lag][0]['ssr_ftest'][1]
        sig  = "✓ Significant" if pval < 0.05 else "✗"
        print(f"    Lag {lag} day: p={pval:.4f}  {sig}")
        granger_results[name][f'US_India_lag{lag}'] = pval

    # India → US
    print(f"  India → US?")
    res = grangercausalitytests(df_scaled[[us_col, india_col]], maxlag=5, verbose=False)
    for lag in [1, 2, 3, 4, 5]:
        pval = res[lag][0]['ssr_ftest'][1]
        sig  = "✓ Significant" if pval < 0.05 else "✗"
        print(f"    Lag {lag} day: p={pval:.4f}  {sig}")
        granger_results[name][f'India_US_lag{lag}'] = pval
# ============================================================
# PLOT — Granger p-values heatmap
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

directions = ['US_to_India', 'India_to_US']
titles     = ['US → India', 'India → US']

for ax, direction, title in zip(axes, directions, titles):
    matrix = pd.DataFrame({
        name: [granger_results[name].get(f'{direction}_lag{l}', np.nan) for l in [1,2,3]]
        for name in ['IT', 'Banking', 'Energy', 'Pharma']
    }, index=['Lag 1', 'Lag 2', 'Lag 3'])

    im = ax.imshow(matrix.values, cmap='RdYlGn_r', vmin=0, vmax=0.1, aspect='auto')
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, fontsize=11)
    ax.set_yticklabels(matrix.index, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    for r in range(len(matrix.index)):
        for c in range(len(matrix.columns)):
            val = matrix.values[r, c]
            if not np.isnan(val):
                color = 'white' if val < 0.05 else 'black'
                ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                        fontsize=10, color=color)

plt.colorbar(im, ax=axes[1], label='p-value (green = significant)')
fig.suptitle("Granger Causality p-values (2018–2024)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("04_granger_causality.png", dpi=300, bbox_inches='tight')
plt.show()
# ============================================================
# SECTOR-WISE ML — No mixing between sectors
# ============================================================

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tscv = TimeSeriesSplit(n_splits=5)

# Each sector is completely independent
sectors = {
    "IT":     {"india": "NIFTY_IT",     "us": "IT_US",     "market_india": "MKT_RF_INDIA", "market_us": "MKT_RF_US"},
    "BANK":   {"india": "NIFTY_BANK",   "us": "BANK_US",   "market_india": "MKT_RF_INDIA", "market_us": "MKT_RF_US"},
    "ENERGY": {"india": "NIFTY_ENERGY", "us": "ENERGY_US", "market_india": "MKT_RF_INDIA", "market_us": "MKT_RF_US"},
    "PHARMA": {"india": "NIFTY_PHARMA", "us": "PHARMA_US", "market_india": "MKT_RF_INDIA", "market_us": "MKT_RF_US"},
}

results = {}

for sector_name, cols in sectors.items():

    print(f"\n{'='*50}")
    print(f"SECTOR: {sector_name}")
    print(f"{'='*50}")

    # Only this sector's data
    features = [cols['market_india'], cols['market_us'], cols['us']]
    target   = cols['india']

    df_sector = monthly[features + [target]].dropna()

    X = df_sector[features].values
    y = df_sector[target].values

    scaler = StandardScaler()
    rf_scores, xgb_scores, lasso_scores = [], [], []
    rf_imp  = np.zeros(len(features))
    xgb_imp = np.zeros(len(features))
    lasso_coefs = np.zeros(len(features))

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale on train only
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Random Forest
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_train_s, y_train)
        rf_scores.append(r2_score(y_test, rf.predict(X_test_s)))
        rf_imp += rf.feature_importances_

        # XGBoost
        xgb = GradientBoostingRegressor(n_estimators=200, random_state=42)
        xgb.fit(X_train_s, y_train)
        xgb_scores.append(r2_score(y_test, xgb.predict(X_test_s)))
        xgb_imp += xgb.feature_importances_

        # LASSO
        lasso = LassoCV(cv=3, random_state=42, max_iter=5000)
        lasso.fit(X_train_s, y_train)
        lasso_scores.append(r2_score(y_test, lasso.predict(X_test_s)))
        lasso_coefs += np.abs(lasso.coef_)

    # Average across folds
    rf_imp      /= tscv.n_splits
    xgb_imp     /= tscv.n_splits
    lasso_coefs /= tscv.n_splits

    results[sector_name] = {
        'features':     features,
        'rf_r2':        np.mean(rf_scores),
        'xgb_r2':       np.mean(xgb_scores),
        'lasso_r2':     np.mean(lasso_scores),
        'rf_imp':       pd.Series(rf_imp,      index=features),
        'xgb_imp':      pd.Series(xgb_imp,     index=features),
        'lasso_coefs':  pd.Series(lasso_coefs, index=features),
    }

    print(f"  Target   : {target}")
    print(f"  Features : {features}")
    print(f"\n  Model Performance (out-of-sample R²):")
    print(f"    LASSO          : {np.mean(lasso_scores):.4f}")
    print(f"    Random Forest  : {np.mean(rf_scores):.4f}")
    print(f"    XGBoost        : {np.mean(xgb_scores):.4f}")
    print(f"\n  Feature Importance (Random Forest):")
    for feat, imp in results[sector_name]['rf_imp'].sort_values(ascending=False).items():
        bar = '█' * int(imp * 40)
        print(f"    {feat:20s}: {imp:.4f}  {bar}")

# ============================================================
# PLOT — One figure per sector, 3 models side by side
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
axes = axes.flatten()

feat_labels = {
    'MKT_RF_INDIA': 'India Market',
    'MKT_RF_US':    'US Market',
    'IT_US':        'US IT',
    'BANK_US':      'US Banking',
    'ENERGY_US':    'US Energy',
    'PHARMA_US':    'US Pharma',
}

bar_colors = {
    'LASSO':         '#2ca02c',
    'Random Forest': '#1f77b4',
    'XGBoost':       '#ff7f0e',
}

for i, (sector_name, res) in enumerate(results.items()):
    ax    = axes[i]
    feats = res['features']
    x     = np.arange(len(feats))
    width = 0.25

    lasso_vals = [res['lasso_coefs'][f] / res['lasso_coefs'].sum() if res['lasso_coefs'].sum() > 0 else 0 for f in feats]
    rf_vals    = [res['rf_imp'][f]      for f in feats]
    xgb_vals   = [res['xgb_imp'][f]    for f in feats]

    ax.bar(x - width, lasso_vals, width, label='LASSO',         color='#2ca02c', alpha=0.85)
    ax.bar(x,         rf_vals,    width, label='Random Forest',  color='#1f77b4', alpha=0.85)
    ax.bar(x + width, xgb_vals,   width, label='XGBoost',        color='#ff7f0e', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([feat_labels.get(f, f) for f in feats], fontsize=10)
    ax.set_ylabel("Importance / Coefficient", fontsize=10)
    ax.set_title(
        f"{sector_name}  |  RF R²={res['rf_r2']:.2f}  XGB R²={res['xgb_r2']:.2f}  LASSO R²={res['lasso_r2']:.2f}",
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4, axis='y')

fig.suptitle("Feature Importance by Sector — LASSO, Random Forest, XGBoost",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("03_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 13. PLOT 3 — FEATURE IMPORTANCE HEATMAP
# ============================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=100)

for i, sector in enumerate(sectors_india):
    df_ml = monthly[[sector] + features].dropna()
    X_scaled = StandardScaler().fit_transform(df_ml[features])
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_scaled, df_ml[sector])
    importances = pd.Series(rf.feature_importances_, index=features).sort_values()

    axes[i].barh(importances.index, importances.values, color='#1f77b4', alpha=0.8)
    axes[i].set_title(sector.replace("NIFTY_", ""), fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Importance", fontsize=9)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

fig.suptitle("Random Forest Feature Importance — Indian Sectors", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("03_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ All done! Saved: 01_monthly_returns.png, 02_rolling_betas.png, 03_feature_importance.png")