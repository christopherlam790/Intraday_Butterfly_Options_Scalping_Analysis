# Intraday_Butterfly_Options_Scalping_Analysis

Analyzing the effectiveness of 0DTE butterfly scalping

---

# Methods

1. Go to Massive.com to et data via online ticket (Copy Paste faster than writiing program to pull & assemble (for free tier))

# Analysis idea

1. Define sticky volatility regimes using GK
2. Within those regimes, identify sticky TA indicators
3. Segment time-of-day inside sticky regimes
4. Re-evaluate indicator stickiness per segment
5. Backtest butterfly pricing across profit targets
6. Benchmark vs structure-appropriate baselines
7. Analyze failure modes explicitly

# Workflow

1. Raw data
   a. json_manipulation.py
   b. json_preprocessing.py
   c. upload_raw_data.py
   d. download_raw_data.py
2. GK sticky regimes
   a. identify_sticky_gk_regimes.py
3. Sticky ta classifications
   a. identify_sticky_ta_classification.py
   b. upload_ta_classification.py
   c. download_ta_classification.py
   d. visualize_sticky_ta_classification.py
4. Black-Scholes Pricing Backtest
