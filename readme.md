Intermarket Sector Analysis: India vs US (2018–2024)
Overview
This project investigates the relationship between major Indian and US equity sectors using traditional econometric methods and machine learning models. The analysis focuses on whether movements in US sector ETFs influence corresponding Indian sectors and how market risk exposures evolve over time.
The study covers:
Information Technology
Banking
Energy
Pharmaceuticals
using data from 2018–2024.

Methodology :
1. Return Construction
Daily returns are used for causality analysis.
Monthly returns are used for factor modeling and machine learning.
Excess returns are computed using a risk-free rate proxy.

2. Factor Regression
Ordinary Least Squares (OLS) regressions estimate:
Alpha
Market Beta
R²
for each sector relative to its domestic market index.

3. Rolling Beta Analysis
A 12-month rolling window regression is used to examine how market sensitivity changes through time, especially during major economic shocks such as:
COVID-19 Crash
Russia–Ukraine Conflict
Federal Reserve Rate Hikes
US Banking Crisis
Indian General Elections

4. Granger Causality
Daily return series are tested for:
US → India influence
India → US influence
across lags ranging from 1–5 trading days.
This helps identify whether information originating in one market statistically precedes movements in the other.

5. Machine Learning Analysis
Three models are trained independently for each sector:
LASSO Regression
Random Forest
Gradient Boosting (XGBoost-style)
Features include:
Indian market excess returns
US market excess returns
Corresponding US sector returns
Model performance is evaluated using out-of-sample R² with Time Series Cross Validation.