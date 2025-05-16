import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_stock_adv(ticker, periods=[30, 60, 90], include_dollar_volume=True):
    """
    Fetch Average Daily Volume (ADV) for a given stock ticker over different time periods.
    
    Args:
        ticker (str): Stock ticker symbol
        periods (list): List of time periods (in days) to calculate ADV for
        include_dollar_volume (bool): Calculate volume in dollar terms as well
    
    Returns:
        dict: Dictionary with ADV values for each period
    """
    # Format Hong Kong stock tickers correctly for yfinance
    original_ticker = ticker
    if ticker.endswith('.HK'):
        # yfinance format: four digits + .HK
        # If ticker is already in correct format, keep it
        pass
    elif ticker.isdigit():
        # Add leading zeros and .HK suffix
        ticker = f"{int(ticker):04d}.HK"
    
    print(f"Using ticker format: {ticker}")
    
    # Get stock data
    stock = yf.Ticker(ticker)
    
    # Calculate end date (today)
    end_date = datetime.now()
    
    # Calculate start date (max period + buffer)
    max_period = max(periods) if periods else 90
    start_date = end_date - timedelta(days=max_period + 10)  # Add buffer for weekends/holidays
    
    # Get historical data
    hist = stock.history(start=start_date, end=end_date)
    
    if hist.empty:
        return {"error": f"No data found for ticker {ticker}"}
    
    # Calculate ADV for each period
    results = {}
    
    # Get current price for dollar volume calculations
    latest_price = hist["Close"].iloc[-1]
    results["latest_price"] = float(latest_price)
    
    # Add most recent volume
    latest_volume = int(hist["Volume"].iloc[-1])
    results["latest_volume"] = latest_volume
    results["latest_date"] = hist.index[-1].strftime("%Y-%m-%d")
    
    # Add dollar volume
    if include_dollar_volume:
        results["latest_dollar_volume"] = latest_volume * latest_price
    
    # Calculate ADVs for different periods
    for period in periods:
        # Make sure we don't try to use more days than we have
        actual_period = min(period, len(hist))
        period_data = hist.tail(actual_period)
        
        # Share volume
        adv = int(period_data["Volume"].mean())
        results[f"ADV_{period}d"] = adv
        
        # Dollar volume
        if include_dollar_volume:
            # Calculate dollar volume for each day then average
            period_data['dollar_volume'] = period_data["Volume"] * period_data["Close"]
            adv_dollar = float(period_data['dollar_volume'].mean())
            results[f"ADV_{period}d_dollar"] = adv_dollar
    
    # Get basic stock info
    try:
        info = stock.info
        results["name"] = info.get("shortName", "N/A")
        results["currency"] = info.get("currency", "N/A")
    except Exception as e:
        results["name"] = ticker
        results["currency"] = "HKD"
        print(f"Warning: Could not fetch additional info: {e}")
    
    return results

def display_results(ticker, results):
    """Display the ADV results in a readable format"""
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    currency = results['currency']
    
    print(f"\n=== Volume Analysis for {results['name']} ({ticker}) ===")
    print(f"Currency: {currency}")
    print(f"Latest Price: {format_currency(results['latest_price'], currency)}")
    
    print(f"\nLatest Trading Date: {results['latest_date']}")
    print(f"Latest Volume (shares): {results['latest_volume']:,}")
    
    if "latest_dollar_volume" in results:
        print(f"Latest Volume (dollar): {format_currency(results['latest_dollar_volume'], currency)}")
    
    print("\nAverage Daily Volumes:")
    
    # Display ADVs - share volume
    for key, value in sorted([i for i in results.items() if i[0].startswith("ADV_") and not i[0].endswith("_dollar")]):
        period = key.split("_")[1].replace("d", " days")
        print(f"{period} (shares): {value:,}")
    
    # Display ADVs - dollar volume
    if any(k.endswith("_dollar") for k in results.keys()):
        print("\nAverage Daily Volumes (in currency terms):")
        for key, value in sorted([i for i in results.items() if i[0].endswith("_dollar")]):
            period = key.split("_")[1].replace("d", " days")
            print(f"{period}: {format_currency(value, currency)}")

def format_currency(value, symbol):
    """Format currency with appropriate suffixes for large numbers"""
    if value >= 1_000_000_000:
        return f"{symbol} {value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{symbol} {value/1_000_000:.2f}M"
    else:
        return f"{symbol} {value:,.2f}"

if __name__ == "__main__":
    # Tencent Holdings - correct yfinance format is 0700.HK
    ticker = "69"
    
    print(f"Fetching volume data for Tencent (ticker {ticker})...")
    results = get_stock_adv(ticker, periods=[20], include_dollar_volume=True)
    display_results(ticker, results) 