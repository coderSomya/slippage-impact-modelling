#!/usr/bin/env python3
"""
Test script for data loading and basic analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_stock_data(symbol, data_dir='Data'):
    """Load all CSV files for a given stock symbol"""
    files = [f for f in os.listdir(data_dir) if f.startswith(symbol) and f.endswith('.csv')]
    
    print(f"Found {len(files)} files for {symbol}")
    
    all_data = []
    for file in files:
        print(f"Loading {file}...")
        df = pd.read_csv(os.path.join(data_dir, file))
        df['symbol'] = symbol
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    """Preprocess the limit order book data"""
    # Convert timestamp
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    
    # Calculate mid price and spread
    df['best_bid'] = df['bid_px_00']
    df['best_ask'] = df['ask_px_00']
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    
    return df

def main():
    print("Market Impact Research - Data Loading Test")
    print("=" * 50)
    
    # Test with one stock first
    symbol = 'FROG'
    print(f"\nLoading {symbol} data...")
    
    try:
        # Load data
        df = load_stock_data(symbol)
        print(f"Loaded {len(df)} records")
        
        # Preprocess
        df = preprocess_data(df)
        print("Preprocessing complete")
        
        # Basic statistics
        print(f"\nBasic Statistics for {symbol}:")
        print(f"Date range: {df['ts_event'].min()} to {df['ts_event'].max()}")
        print(f"Average mid price: ${df['mid_price'].mean():.2f}")
        print(f"Average spread: ${df['spread'].mean():.4f}")
        print(f"Min spread: ${df['spread'].min():.4f}")
        print(f"Max spread: ${df['spread'].max():.4f}")
        
        # Sample data
        print(f"\nSample data:")
        sample_cols = ['ts_event', 'best_bid', 'best_ask', 'mid_price', 'spread', 'symbol']
        print(df[sample_cols].head())
        
        print("\nData loading test successful!")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 