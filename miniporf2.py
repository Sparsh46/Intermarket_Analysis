# ============================================================
# GRANGER CAUSALITY — DAILY — BOTH DIRECTIONS
# ============================================================
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ============================================================
# DATA COLLECTION
# ============================================================
tickers = {
    "NIFTY_IT":     "^CNXIT",
    "NIFTY_BANK":   "^NSEBANK",
    "NIFTY_ENERGY": "^CNXENERGY",
    "NIFTY_PHARMA": "^CNXPHARMA",
    "IT_US":        "XLK",
    "BANK_US":      "XLF",
    "ENERGY_US":    "XLE",
    "PHARMA_US":    "XLV",
}

raw = yf.download(list(tickers.values()), start="2018-01-01", end="2024-12-31")["Close"]
raw.columns = list(tickers.keys())

# Daily returns
daily = raw.pct_change().dropna(how='any')

scaler = StandardScaler()

granger_pairs = [
    ("NIFTY_IT",     "IT_US",     "IT"),
    ("NIFTY_BANK",   "BANK_US",   "Banking"),
    ("NIFTY_ENERGY", "ENERGY_US", "Energy"),
    ("NIFTY_PHARMA", "PHARMA_US", "Pharma"),
]

granger_results = {}

print("="*60)
print("ADF STATIONARITY CHECK — DAILY")
print("="*60)
for india_col, us_col, name in granger_pairs:
    for col in [india_col, us_col]:
        pval = adfuller(daily[col].dropna())[1]
        status = "✓ Stationary" if pval < 0.05 else "✗ Needs differencing"
        print(f"  {col:20s}: p={pval:.4f}  {status}")

print("\n" + "="*60)
print("GRANGER CAUSALITY — DAILY (lag = 1 to 5 days)")
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

    print(f"\n{'─'*40}")
    print(f"SECTOR: {name}")
    print(f"{'─'*40}")

    # ── Direction 1: US → India ──
    print(f"  US → India ({us_col} → {india_col})?")
    res_us_india = grangercausalitytests(
        df_scaled[[india_col, us_col]],
        maxlag=5, verbose=False
    )
    for lag in range(1, 6):
        pval = res_us_india[lag][0]['ssr_ftest'][1]
        sig  = "✓ Significant" if pval < 0.05 else "✗"
        print(f"    Lag {lag} day: p={pval:.4f}  {sig}")
        granger_results[name][f'US_India_lag{lag}'] = pval

    # ── Direction 2: India → US ──
    print(f"  India → US ({india_col} → {us_col})?")
    res_india_us = grangercausalitytests(
        df_scaled[[us_col, india_col]],
        maxlag=5, verbose=False
    )
    for lag in range(1, 6):
        pval = res_india_us[lag][0]['ssr_ftest'][1]
        sig  = "✓ Significant" if pval < 0.05 else "✗"
        print(f"    Lag {lag} day: p={pval:.4f}  {sig}")
        granger_results[name][f'India_US_lag{lag}'] = pval

# ============================================================
# PLOT — Heatmap both directions side by side
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

directions = [
    ('US_India', 'US → India'),
    ('India_US', 'India → US'),
]

sector_names = ['IT', 'Banking', 'Energy', 'Pharma']
lags         = [1, 2, 3, 4, 5]

for ax, (direction_key, direction_label) in zip(axes, directions):

    matrix = pd.DataFrame(
        {name: [granger_results[name][f'{direction_key}_lag{lag}'] for lag in lags]
         for name in sector_names},
        index=[f'Lag {l}d' for l in lags]
    )

    im = ax.imshow(matrix.values, cmap='RdYlGn_r', vmin=0, vmax=0.15, aspect='auto')

    ax.set_xticks(range(len(sector_names)))
    ax.set_yticks(range(len(lags)))
    ax.set_xticklabels(sector_names, fontsize=11)
    ax.set_yticklabels([f'Lag {l}d' for l in lags], fontsize=10)
    ax.set_title(direction_label, fontsize=13, fontweight='bold', pad=12)

    # Annotate p-values
    for r in range(len(lags)):
        for c in range(len(sector_names)):
            val = matrix.values[r, c]
            color = 'white' if val < 0.05 else 'black'
            sig   = '✓' if val < 0.05 else ''
            ax.text(c, r, f'{val:.3f}\n{sig}',
                    ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

plt.colorbar(im, ax=axes[1], label='p-value  (green = significant < 0.05)')
fig.suptitle("Granger Causality — Daily Returns (2018–2024)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("04_granger_causality.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("SUMMARY — SIGNIFICANT CAUSALITY (p < 0.05)")
print("="*60)
print(f"{'Sector':<12} {'US→India':<30} {'India→US':<30}")
print("─"*70)

for name in sector_names:
    us_india_lags   = [l for l in range(1,6) if granger_results[name][f'US_India_lag{l}'] < 0.05]
    india_us_lags   = [l for l in range(1,6) if granger_results[name][f'India_US_lag{l}'] < 0.05]

    us_india_str  = f"Sig at lags {us_india_lags}"   if us_india_lags   else "Not significant"
    india_us_str  = f"Sig at lags {india_us_lags}"   if india_us_lags   else "Not significant"

    print(f"{name:<12} {us_india_str:<30} {india_us_str:<30}")