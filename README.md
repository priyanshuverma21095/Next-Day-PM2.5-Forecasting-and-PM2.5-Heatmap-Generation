# PM2.5 Forecasting and Heatmaps (CS771 Mini-Project 1)

Next-day (24 h lead) PM2.5 forecasting at 468 low-cost sensor stations in Bihar,
inverse-distance-weighted (IDW) forecast maps, and cleaning of a raw 76-sensor
Kanpur dataset. Built for CS771 (Autumn 2026) at IIT Kanpur.

## What's inside
- **Part 0 – Kanpur data cleaning:** hourly reshaping, flags for error codes,
  zeros, stuck values, unusable sensors and network outages, and a cross-device
  calibration check. No gap filling.
- **Part 1 – Baselines:** persistence and a 29-feature Ridge model
  (validation RMSE 32.75 vs 29.58 µg/m³; persistence wins on MAE).
- **Part 2 – Improved model:** feature groups with an ablation study,
  tuned on validation only.
- **Part 3 – Lead time:** 1, 3, 6, 12 and 24 h leads, and the 24 h persistence dip
  explained by the daily cycle.
- **Part 4 – Maps:** IDW forecast vs observed maps, map RMSE and pattern correlation.
- **Part 5 – `forecast.py`:** the submission function `forecast(history, issue_time)`,
  which loads `model.pkl` and never refits.

## Rules followed
Time-based train/validation split (no leakage), no external data,
no neural networks, fixed random seeds.

## Run it
Put the course data in `data/`, then run the notebooks top to bottom.
Check the submission with: `python starter/check_submission.py submission`.
