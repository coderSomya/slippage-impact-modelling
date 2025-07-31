"""
Comprehensive Exploratory Data Analysis for Market Impact Research

This script performs exhaustive EDA on limit order book data to understand:
- Market microstructure patterns
- Order book dynamics
- Price and volume distributions
- Time-series characteristics
- Cross-sectional variations across stocks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy import stats
from scipy.stats import kurtosis, skew
import warnings
warnings.filterwarnings('ignore')


import os
os.makedirs('plots/eda', exist_ok=True)
os.makedirs('plots/academic_research', exist_ok=True)

# Set plotting style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_and_preprocess_data(symbol, data_dir='Data'):
    """Load and preprocess data for a single stock with comprehensive feature engineering"""
    
    files = [f for f in os.listdir(data_dir) if f.startswith(symbol) and f.endswith('.csv')]
    all_data = []
    
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file))
        df['symbol'] = symbol
        all_data.append(df)
    
    df = pd.concat(all_data, ignore_index=True)
    df['ts_event'] = pd.to_datetime(df['ts_event'], format='mixed')
    
    # Core price features
    df['best_bid'] = df['bid_px_00']
    df['best_ask'] = df['ask_px_00']
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000  # Basis points
    
    # Order book depth features
    bid_sz_cols = [col for col in df.columns if col.startswith('bid_sz_')]
    ask_sz_cols = [col for col in df.columns if col.startswith('ask_sz_')]
    df['bid_depth'] = df[bid_sz_cols].sum(axis=1)
    df['ask_depth'] = df[ask_sz_cols].sum(axis=1)
    df['total_depth'] = df['bid_depth'] + df['ask_depth']
    
    # Price level features (up to 5 levels)
    for i in range(5):
        df[f'bid_px_{i}'] = df[f'bid_px_{i:02d}']
        df[f'ask_px_{i}'] = df[f'ask_px_{i:02d}']
        df[f'bid_sz_{i}'] = df[f'bid_sz_{i:02d}']
        df[f'ask_sz_{i}'] = df[f'ask_sz_{i:02d}']
    
    # Market microstructure features
    df['bid_ask_imbalance'] = (df['bid_depth'] - df['ask_depth']) / df['total_depth']
    df['price_pressure'] = df['bid_ask_imbalance'] * df['spread_bps']
    
    # Time features
    df['hour'] = df['ts_event'].dt.hour
    df['minute'] = df['ts_event'].dt.minute
    df['day_of_week'] = df['ts_event'].dt.dayofweek
    df['is_market_open'] = ((df['hour'] >= 9) & (df['hour'] < 16)) | ((df['hour'] == 16) & (df['minute'] <= 30))
    
    return df

def analyze_price_distributions(df, symbol):
    """Analyze price distributions and their statistical properties"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{symbol} - Price Distribution Analysis', fontsize=16)
    
    # Mid price distribution
    axes[0,0].hist(df['mid_price'], bins=50, alpha=0.7, color='blue')
    axes[0,0].set_title('Mid Price Distribution')
    axes[0,0].set_xlabel('Price ($)')
    axes[0,0].set_ylabel('Frequency')
    
    # Spread distribution
    axes[0,1].hist(df['spread'], bins=50, alpha=0.7, color='green')
    axes[0,1].set_title('Spread Distribution')
    axes[0,1].set_xlabel('Spread ($)')
    axes[0,1].set_ylabel('Frequency')
    
    # Spread in basis points
    axes[0,2].hist(df['spread_bps'], bins=50, alpha=0.7, color='red')
    axes[0,2].set_title('Spread Distribution (Basis Points)')
    axes[0,2].set_xlabel('Spread (bps)')
    axes[0,2].set_ylabel('Frequency')
    
    # Log returns of mid price
    log_returns = np.log(df['mid_price'] / df['mid_price'].shift(1)).dropna()
    axes[1,0].hist(log_returns, bins=50, alpha=0.7, color='purple')
    axes[1,0].set_title('Log Returns Distribution')
    axes[1,0].set_xlabel('Log Returns')
    axes[1,0].set_ylabel('Frequency')
    
    # Q-Q plot for log returns
    stats.probplot(log_returns, dist="norm", plot=axes[1,1])
    axes[1,1].set_title('Q-Q Plot (Log Returns vs Normal)')
    
    # Price level analysis
    price_levels = []
    for i in range(5):
        price_levels.extend([df[f'bid_px_{i}'].mean(), df[f'ask_px_{i}'].mean()])
    
    axes[1,2].bar(range(len(price_levels)), price_levels, alpha=0.7)
    axes[1,2].set_title('Average Price Levels')
    axes[1,2].set_xlabel('Price Level')
    axes[1,2].set_ylabel('Price ($)')
    
    plt.tight_layout()
    plt.savefig(f'plots/eda/{symbol}_price_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    stats_summary = {
        'mid_price_mean': df['mid_price'].mean(),
        'mid_price_std': df['mid_price'].std(),
        'mid_price_skew': skew(df['mid_price']),
        'mid_price_kurtosis': kurtosis(df['mid_price']),
        'spread_mean': df['spread'].mean(),
        'spread_std': df['spread'].std(),
        'spread_bps_mean': df['spread_bps'].mean(),
        'log_returns_std': log_returns.std(),
        'log_returns_skew': skew(log_returns),
        'log_returns_kurtosis': kurtosis(log_returns)
    }
    
    return stats_summary

def analyze_order_book_dynamics(df, symbol):
    """Analyze order book depth and dynamics"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{symbol} - Order Book Dynamics', fontsize=16)
    
    # Depth distribution
    axes[0,0].hist(df['total_depth'], bins=50, alpha=0.7, color='blue')
    axes[0,0].set_title('Total Order Book Depth')
    axes[0,0].set_xlabel('Depth (shares)')
    axes[0,0].set_ylabel('Frequency')
    
    # Bid vs Ask depth
    axes[0,1].scatter(df['bid_depth'], df['ask_depth'], alpha=0.5, s=1)
    axes[0,1].set_title('Bid vs Ask Depth')
    axes[0,1].set_xlabel('Bid Depth')
    axes[0,1].set_ylabel('Ask Depth')
    
    # Bid-ask imbalance
    axes[0,2].hist(df['bid_ask_imbalance'], bins=50, alpha=0.7, color='green')
    axes[0,2].set_title('Bid-Ask Imbalance Distribution')
    axes[0,2].set_xlabel('Imbalance')
    axes[0,2].set_ylabel('Frequency')
    
    # Size distribution at different levels
    level_sizes = []
    level_labels = []
    for i in range(5):
        level_sizes.extend([df[f'bid_sz_{i}'].mean(), df[f'ask_sz_{i}'].mean()])
        level_labels.extend([f'Bid_{i}', f'Ask_{i}'])
    
    axes[1,0].bar(range(len(level_sizes)), level_sizes, alpha=0.7)
    axes[1,0].set_title('Average Size by Level')
    axes[1,0].set_xlabel('Level')
    axes[1,0].set_ylabel('Average Size')
    axes[1,0].set_xticks(range(len(level_labels)))
    axes[1,0].set_xticklabels(level_labels, rotation=45)
    
    # Price pressure
    axes[1,1].hist(df['price_pressure'], bins=50, alpha=0.7, color='red')
    axes[1,1].set_title('Price Pressure Distribution')
    axes[1,1].set_xlabel('Price Pressure')
    axes[1,1].set_ylabel('Frequency')
    
    # Depth vs Spread relationship
    axes[1,2].scatter(df['total_depth'], df['spread_bps'], alpha=0.5, s=1)
    axes[1,2].set_title('Depth vs Spread')
    axes[1,2].set_xlabel('Total Depth')
    axes[1,2].set_ylabel('Spread (bps)')
    
    plt.tight_layout()
    plt.savefig(f'plots/eda/{symbol}_order_book_dynamics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'avg_total_depth': df['total_depth'].mean(),
        'avg_bid_depth': df['bid_depth'].mean(),
        'avg_ask_depth': df['ask_depth'].mean(),
        'depth_std': df['total_depth'].std(),
        'imbalance_mean': df['bid_ask_imbalance'].mean(),
        'imbalance_std': df['bid_ask_imbalance'].std(),
        'price_pressure_mean': df['price_pressure'].mean(),
        'price_pressure_std': df['price_pressure'].std()
    }

def analyze_time_series_patterns(df, symbol):
    """Analyze time-series patterns and market microstructure"""
    
    # Resample to 1-minute intervals for time series analysis
    df_resampled = df.set_index('ts_event').resample('1T').agg({
        'mid_price': 'mean',
        'spread': 'mean',
        'spread_bps': 'mean',
        'total_depth': 'mean',
        'bid_ask_imbalance': 'mean',
        'price_pressure': 'mean'
    }).dropna()
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle(f'{symbol} - Time Series Analysis', fontsize=16)
    
    # Mid price time series
    axes[0,0].plot(df_resampled.index, df_resampled['mid_price'], linewidth=0.5)
    axes[0,0].set_title('Mid Price Time Series')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Price ($)')
    
    # Spread time series
    axes[0,1].plot(df_resampled.index, df_resampled['spread_bps'], linewidth=0.5, color='red')
    axes[0,1].set_title('Spread Time Series (bps)')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Spread (bps)')
    
    # Depth time series
    axes[1,0].plot(df_resampled.index, df_resampled['total_depth'], linewidth=0.5, color='green')
    axes[1,0].set_title('Total Depth Time Series')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Depth')
    
    # Imbalance time series
    axes[1,1].plot(df_resampled.index, df_resampled['bid_ask_imbalance'], linewidth=0.5, color='purple')
    axes[1,1].set_title('Bid-Ask Imbalance Time Series')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Imbalance')
    
    # Intraday patterns
    hourly_stats = df.groupby('hour').agg({
        'spread_bps': 'mean',
        'total_depth': 'mean',
        'bid_ask_imbalance': 'mean'
    })
    
    axes[2,0].plot(hourly_stats.index, hourly_stats['spread_bps'], marker='o')
    axes[2,0].set_title('Intraday Spread Pattern')
    axes[2,0].set_xlabel('Hour')
    axes[2,0].set_ylabel('Average Spread (bps)')
    
    axes[2,1].plot(hourly_stats.index, hourly_stats['total_depth'], marker='o', color='green')
    axes[2,1].set_title('Intraday Depth Pattern')
    axes[2,1].set_xlabel('Hour')
    axes[2,1].set_ylabel('Average Depth')
    
    plt.tight_layout()
    plt.savefig(f'plots/eda/{symbol}_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'price_volatility': df_resampled['mid_price'].pct_change().std(),
        'spread_volatility': df_resampled['spread_bps'].std(),
        'depth_volatility': df_resampled['total_depth'].pct_change().std(),
        'autocorr_price': df_resampled['mid_price'].autocorr(),
        'autocorr_spread': df_resampled['spread_bps'].autocorr(),
        'autocorr_depth': df_resampled['total_depth'].autocorr()
    }

def analyze_cross_sectional_variations(symbols_data):
    """Compare characteristics across different stocks"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Cross-Sectional Analysis Across Stocks', fontsize=16)
    
    # Average prices
    avg_prices = [symbols_data[s]['price_stats']['mid_price_mean'] for s in symbols_data.keys()]
    axes[0,0].bar(symbols_data.keys(), avg_prices, alpha=0.7)
    axes[0,0].set_title('Average Mid Prices')
    axes[0,0].set_ylabel('Price ($)')
    
    # Average spreads
    avg_spreads = [symbols_data[s]['price_stats']['spread_bps_mean'] for s in symbols_data.keys()]
    axes[0,1].bar(symbols_data.keys(), avg_spreads, alpha=0.7, color='red')
    axes[0,1].set_title('Average Spreads (bps)')
    axes[0,1].set_ylabel('Spread (bps)')
    
    # Price volatility
    price_vols = [symbols_data[s]['time_stats']['price_volatility'] for s in symbols_data.keys()]
    axes[0,2].bar(symbols_data.keys(), price_vols, alpha=0.7, color='green')
    axes[0,2].set_title('Price Volatility')
    axes[0,2].set_ylabel('Volatility')
    
    # Average depth
    avg_depths = [symbols_data[s]['orderbook_stats']['avg_total_depth'] for s in symbols_data.keys()]
    axes[1,0].bar(symbols_data.keys(), avg_depths, alpha=0.7, color='purple')
    axes[1,0].set_title('Average Total Depth')
    axes[1,0].set_ylabel('Depth')
    
    # Depth volatility
    depth_vols = [symbols_data[s]['time_stats']['depth_volatility'] for s in symbols_data.keys()]
    axes[1,1].bar(symbols_data.keys(), depth_vols, alpha=0.7, color='orange')
    axes[1,1].set_title('Depth Volatility')
    axes[1,1].set_ylabel('Volatility')
    
    # Imbalance
    imbalances = [symbols_data[s]['orderbook_stats']['imbalance_mean'] for s in symbols_data.keys()]
    axes[1,2].bar(symbols_data.keys(), imbalances, alpha=0.7, color='brown')
    axes[1,2].set_title('Average Bid-Ask Imbalance')
    axes[1,2].set_ylabel('Imbalance')
    
    plt.tight_layout()
    plt.savefig('plots/eda/cross_sectional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(symbols_data):
    """Generate a comprehensive EDA report"""
    
    print("=" * 80)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 80)
    
    print("\n1. DATA OVERVIEW")
    print("-" * 40)
    for symbol, data in symbols_data.items():
        print(f"\n{symbol}:")
        print(f"  Records: {data['n_records']:,}")
        print(f"  Date Range: {data['date_range']}")
        print(f"  Average Price: ${data['price_stats']['mid_price_mean']:.2f}")
        print(f"  Average Spread: {data['price_stats']['spread_bps_mean']:.2f} bps")
        print(f"  Price Volatility: {data['time_stats']['price_volatility']:.4f}")
        print(f"  Average Depth: {data['orderbook_stats']['avg_total_depth']:,.0f}")

def main():
    """Main EDA execution"""
    
    symbols = ['FROG', 'CRWV', 'SOUN']
    symbols_data = {}
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        # Load and preprocess
        df = load_and_preprocess_data(symbol)
        
        # Perform analyses
        price_stats = analyze_price_distributions(df, symbol)
        orderbook_stats = analyze_order_book_dynamics(df, symbol)
        time_stats = analyze_time_series_patterns(df, symbol)
        
        # Store results
        symbols_data[symbol] = {
            'n_records': len(df),
            'date_range': f"{df['ts_event'].min()} to {df['ts_event'].max()}",
            'price_stats': price_stats,
            'orderbook_stats': orderbook_stats,
            'time_stats': time_stats
        }
    
    # Cross-sectional analysis
    analyze_cross_sectional_variations(symbols_data)
    
    # Generate report
    generate_comprehensive_report(symbols_data)

if __name__ == "__main__":
    main() 