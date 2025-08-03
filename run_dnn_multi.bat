@echo off
cd /d D:\repo\etf_forecast
py .\pca_features_daily.py
py .\main_dnn_multi.py
@REM pause