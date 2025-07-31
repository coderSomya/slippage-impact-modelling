"""
Market Impact Research: Temporary Impact Function Modeling

This script conducts comprehensive research on market impact modeling and optimal order allocation.
It addresses the problem of modeling temporary impact functions and formulating strategies to minimize
total impact when executing large orders.

Problem Statement:
1. Model the temporary impact function g_t(X) that describes slippage when placing X orders at time t
2. Formulate a mathematical framework/algorithm for optimal order allocation over N=390 trading periods

Data: Limit order book data for 3 stocks (FROG, CRWV, SOUN) with high-frequency snapshots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Ensure plot directories exist
import os
os.makedirs('plots/eda', exist_ok=True)
os.makedirs('plots/academic_research', exist_ok=True)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

class MarketImpactResearch:
    """
    Comprehensive market impact research class that handles data loading, analysis, 
    model fitting, and optimization strategy development.
    """
    
    def __init__(self, data_dir='Data'):
        self.data_dir = data_dir
        self.stock_data = {}
        self.impact_results = {}
        self.model_results = {}
        
    def load_stock_data(self, symbol):
        """
        Load all CSV files for a given stock symbol.
        
        Args:
            symbol (str): Stock symbol (FROG, CRWV, SOUN)
            
        Returns:
            pd.DataFrame: Combined data for the stock
        """
        files = [f for f in os.listdir(self.data_dir) if f.startswith(symbol) and f.endswith('.csv')]
        
        print(f"Loading {len(files)} files for {symbol}...")
        
        all_data = []
        for file in files:
            print(f"  Loading {file}...")
            df = pd.read_csv(os.path.join(self.data_dir, file))
            df['symbol'] = symbol
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"  Total records: {len(combined_df):,}")
        return combined_df
    
    def preprocess_data(self, df):
        """
        Preprocess the limit order book data.
        
        Args:
            df (pd.DataFrame): Raw limit order book data
            
        Returns:
            pd.DataFrame: Preprocessed data with calculated features
        """
        print("  Preprocessing data...")
        
        # Convert timestamp with mixed format handling
        df['ts_event'] = pd.to_datetime(df['ts_event'], format='mixed')
        
        # Calculate mid price and spread
        df['best_bid'] = df['bid_px_00']
        df['best_ask'] = df['ask_px_00']
        df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
        df['spread'] = df['best_ask'] - df['best_bid']
        
        # Calculate order book depth
        bid_sz_cols = [col for col in df.columns if col.startswith('bid_sz_')]
        ask_sz_cols = [col for col in df.columns if col.startswith('ask_sz_')]
        df['bid_depth'] = df[bid_sz_cols].sum(axis=1)
        df['ask_depth'] = df[ask_sz_cols].sum(axis=1)
        
        # Basic statistics
        print(f"    Date range: {df['ts_event'].min()} to {df['ts_event'].max()}")
        print(f"    Average mid price: ${df['mid_price'].mean():.2f}")
        print(f"    Average spread: ${df['spread'].mean():.4f}")
        
        return df
    
    def calculate_temp_impact(self, row, order_size, side='buy'):
        """
        Calculate temporary impact for a given order size by simulating order execution
        against the limit order book.
        
        Args:
            row (pd.Series): Single row of limit order book data
            order_size (int): Number of shares to trade
            side (str): 'buy' or 'sell'
            
        Returns:
            float: Temporary impact (slippage) in dollars
        """
        if side == 'buy':
            # Buy side - impact on ask prices
            ask_prices = [row[f'ask_px_{i:02d}'] for i in range(10)]
            ask_sizes = [row[f'ask_sz_{i:02d}'] for i in range(10)]
            
            remaining_size = order_size
            total_cost = 0
            
            for i in range(len(ask_prices)):
                if remaining_size <= 0:
                    break
                
                executed_size = min(remaining_size, ask_sizes[i])
                total_cost += executed_size * ask_prices[i]
                remaining_size -= executed_size
            
            if order_size > 0:
                avg_price = total_cost / order_size
                impact = avg_price - row['mid_price']
                return impact
            return 0
        
        else:
            # Sell side - impact on bid prices
            bid_prices = [row[f'bid_px_{i:02d}'] for i in range(10)]
            bid_sizes = [row[f'bid_sz_{i:02d}'] for i in range(10)]
            
            remaining_size = order_size
            total_cost = 0
            
            for i in range(len(bid_prices)):
                if remaining_size <= 0:
                    break
                
                executed_size = min(remaining_size, bid_sizes[i])
                total_cost += executed_size * bid_prices[i]
                remaining_size -= executed_size
            
            if order_size > 0:
                avg_price = total_cost / order_size
                impact = row['mid_price'] - avg_price
                return impact
            return 0
    
    def analyze_impact_patterns(self, df, symbol, sample_size=1000):
        """
        Analyze temporary impact patterns for different order sizes.
        
        Args:
            df (pd.DataFrame): Stock data
            symbol (str): Stock symbol
            sample_size (int): Number of data points to sample
            
        Returns:
            tuple: (order_sizes, buy_impacts, sell_impacts)
        """
        print(f"  Analyzing impact patterns for {symbol}...")
        
        # Sample data points for efficiency
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        order_sizes = np.array([100, 500, 1000, 2000, 5000, 10000])
        impacts_buy = []
        impacts_sell = []
        
        for size in order_sizes:
            buy_impacts = []
            sell_impacts = []
            
            for _, row in sample_df.iterrows():
                buy_impact = self.calculate_temp_impact(row, size, 'buy')
                sell_impact = self.calculate_temp_impact(row, size, 'sell')
                
                if buy_impact > 0:  # Valid impact
                    buy_impacts.append(buy_impact)
                if sell_impact > 0:  # Valid impact
                    sell_impacts.append(sell_impact)
            
            impacts_buy.append(np.mean(buy_impacts) if buy_impacts else 0)
            impacts_sell.append(np.mean(sell_impacts) if sell_impacts else 0)
        
        print(f"    Analyzed {len(order_sizes)} order sizes")
        return order_sizes, np.array(impacts_buy), np.array(impacts_sell)
    
    def fit_impact_models(self, sizes, impacts):
        """
        Fit different impact models to the data.
        
        Args:
            sizes (np.array): Order sizes
            impacts (np.array): Corresponding impacts
            
        Returns:
            dict: Model results with R² scores and predictions
        """
        print("  Fitting impact models...")
        
        # Linear model: g(x) = β * x
        linear_model = LinearRegression()
        linear_model.fit(sizes.reshape(-1, 1), impacts)
        linear_pred = linear_model.predict(sizes.reshape(-1, 1))
        linear_r2 = r2_score(impacts, linear_pred)
        
        # Square root model: g(x) = α * √x
        sqrt_sizes = np.sqrt(sizes)
        sqrt_model = LinearRegression()
        sqrt_model.fit(sqrt_sizes.reshape(-1, 1), impacts)
        sqrt_pred = sqrt_model.predict(sqrt_sizes.reshape(-1, 1))
        sqrt_r2 = r2_score(impacts, sqrt_pred)
        
        # Power law model: g(x) = γ * x^p
        powers = [0.3, 0.5, 0.7, 0.8, 0.9]
        power_results = {}
        
        for p in powers:
            power_sizes = sizes ** p
            power_model = LinearRegression()
            power_model.fit(power_sizes.reshape(-1, 1), impacts)
            power_pred = power_model.predict(power_sizes.reshape(-1, 1))
            power_r2 = r2_score(impacts, power_pred)
            power_results[p] = {
                'model': power_model,
                'r2': power_r2,
                'pred': power_pred
            }
        
        best_power = max(power_results.keys(), key=lambda p: power_results[p]['r2'])
        
        print(f"    Linear R²: {linear_r2:.4f}")
        print(f"    Square Root R²: {sqrt_r2:.4f}")
        print(f"    Power Law (p={best_power:.1f}) R²: {power_results[best_power]['r2']:.4f}")
        
        return {
            'linear': {'model': linear_model, 'r2': linear_r2, 'pred': linear_pred},
            'sqrt': {'model': sqrt_model, 'r2': sqrt_r2, 'pred': sqrt_pred},
            'power': power_results[best_power],
            'best_power': best_power
        }
    
    def run_complete_analysis(self, symbols=['FROG', 'CRWV', 'SOUN']):
        """
        Run complete analysis for all symbols.
        
        Args:
            symbols (list): List of stock symbols to analyze
        """
        print("=" * 60)
        print("MARKET IMPACT RESEARCH - COMPLETE ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        print("\n1. DATA LOADING AND PREPROCESSING")
        print("-" * 40)
        
        for symbol in symbols:
            print(f"\nProcessing {symbol}...")
            self.stock_data[symbol] = self.load_stock_data(symbol)
            self.stock_data[symbol] = self.preprocess_data(self.stock_data[symbol])
        
        # Step 2: Analyze impact patterns
        print("\n2. IMPACT PATTERN ANALYSIS")
        print("-" * 40)
        
        for symbol in symbols:
            sizes, buy_impacts, sell_impacts = self.analyze_impact_patterns(
                self.stock_data[symbol], symbol
            )
            self.impact_results[symbol] = {
                'sizes': sizes,
                'buy_impacts': buy_impacts,
                'sell_impacts': sell_impacts
            }
        
        # Step 3: Fit models
        print("\n3. MODEL FITTING")
        print("-" * 40)
        
        for symbol in symbols:
            print(f"\nModeling {symbol} Buy Impact:")
            buy_models = self.fit_impact_models(
                self.impact_results[symbol]['sizes'], 
                self.impact_results[symbol]['buy_impacts']
            )
            self.model_results[symbol] = buy_models
        
        # Step 4: Generate visualizations
        self.create_visualizations()
        
        # Step 5: Print comprehensive summary
        self.print_comprehensive_summary()
        
        # Step 6: Demonstrate optimization framework
        self.demonstrate_optimization()
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the analysis."""
        print("\n4. CREATING VISUALIZATIONS")
        print("-" * 40)
        
        # Plot 1: Impact vs Order Size
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, symbol in enumerate(['FROG', 'CRWV', 'SOUN']):
            sizes = self.impact_results[symbol]['sizes']
            buy_impacts = self.impact_results[symbol]['buy_impacts']
            sell_impacts = self.impact_results[symbol]['sell_impacts']
            
            axes[i].plot(sizes, buy_impacts, 'o-', label='Buy Impact', linewidth=2, markersize=8)
            axes[i].plot(sizes, sell_impacts, 's-', label='Sell Impact', linewidth=2, markersize=8)
            axes[i].set_xlabel('Order Size')
            axes[i].set_ylabel('Temporary Impact ($)')
            axes[i].set_title(f'{symbol} - Temporary Impact vs Order Size')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/academic_research/impact_analysis.png', dpi=300, bbox_inches='tight')
        print("  Saved: plots/academic_research/impact_analysis.png")
        
        # Plot 2: Model Comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, symbol in enumerate(['FROG', 'CRWV', 'SOUN']):
            sizes = self.impact_results[symbol]['sizes']
            impacts = self.impact_results[symbol]['buy_impacts']
            models = self.model_results[symbol]
            
            axes[i].scatter(sizes, impacts, color='blue', alpha=0.7, label='Actual Data', s=100)
            axes[i].plot(sizes, models['linear']['pred'], 'r-', linewidth=2,
                        label=f"Linear (R²={models['linear']['r2']:.3f})")
            axes[i].plot(sizes, models['sqrt']['pred'], 'g-', linewidth=2,
                        label=f"Square Root (R²={models['sqrt']['r2']:.3f})")
            axes[i].plot(sizes, models['power']['pred'], 'orange', linewidth=2,
                        label=f"Power Law p={models['best_power']:.1f} (R²={models['power']['r2']:.3f})")
            
            axes[i].set_xlabel('Order Size')
            axes[i].set_ylabel('Temporary Impact ($)')
            axes[i].set_title(f'{symbol} - Model Comparison')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/academic_research/model_comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: plots/academic_research/model_comparison.png")
        
        plt.show()
    
    def print_comprehensive_summary(self):
        """Print a comprehensive summary of the research findings."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE RESEARCH SUMMARY")
        print("=" * 60)
        
        print("\n1. TEMPORARY IMPACT FUNCTION MODELING")
        print("-" * 40)
        
        for symbol in ['FROG', 'CRWV', 'SOUN']:
            models = self.model_results[symbol]
            print(f"\n{symbol}:")
            print(f"  • Linear Model R²: {models['linear']['r2']:.4f}")
            print(f"  • Square Root Model R²: {models['sqrt']['r2']:.4f}")
            print(f"  • Power Law Model R²: {models['power']['r2']:.4f} (p={models['best_power']:.1f})")
    
    def demonstrate_optimization(self):
        """Demonstrate the optimization framework with a simple example."""
        print("\n6. OPTIMIZATION FRAMEWORK DEMONSTRATION")
        print("-" * 40)
        
        class ImpactOptimizer:
            def __init__(self, total_shares, num_periods=390):
                self.total_shares = total_shares
                self.num_periods = num_periods
            
            def power_impact(self, x, gamma, p):
                """Power law impact model"""
                return gamma * (x ** p)
            
            def objective_function(self, x, impact_params):
                """Total impact to minimize"""
                total_impact = 0
                for i in range(self.num_periods):
                    total_impact += self.power_impact(x[i], impact_params[i]['gamma'], impact_params[i]['p'])
                return total_impact
            
            def constraint_function(self, x):
                """Constraint: sum must equal total shares"""
                return np.sum(x) - self.total_shares
            
            def optimize(self, impact_params):
                """Optimize order allocation"""
                # Initial guess: equal allocation
                x0 = np.ones(self.num_periods) * (self.total_shares / self.num_periods)
                
                # Bounds: non-negative allocations
                bounds = [(0, None)] * self.num_periods
                
                # Constraints
                constraints = [{'type': 'eq', 'fun': self.constraint_function}]
                
                # Optimization
                result = minimize(
                    lambda x: self.objective_function(x, impact_params),
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000}
                )
                
                return result
        
        # Example optimization
        print("Example: Optimizing order allocation for 10,000 shares")
        
        # Sample impact parameters (simplified)
        impact_params = []
        for i in range(390):
            base_impact = 0.001
            time_factor = 1 + 0.5 * np.sin(2 * np.pi * i / 390)
            impact_params.append({
                'gamma': base_impact * time_factor,
                'p': 0.7
            })
        
        optimizer = ImpactOptimizer(10000)
        result = optimizer.optimize(impact_params)
        
        if result.success:
            print(f"✓ Optimization successful!")
            print(f"  Total impact: ${result.fun:.4f}")
            print(f"  Allocation sum: {np.sum(result.x):.2f}")
            print(f"  Min allocation: {np.min(result.x):.2f}")
            print(f"  Max allocation: {np.max(result.x):.2f}")
            print(f"  Average allocation: {np.mean(result.x):.2f}")
        else:
            print(f"✗ Optimization failed: {result.message}")
        
def main():
    """Main function to run the complete market impact research."""
    
    print("Starting Market Impact Research...")
    print("This will analyze temporary impact functions and develop optimization strategies.")
    
    # Create research instance
    researcher = MarketImpactResearch()
    
    # Run complete analysis
    researcher.run_complete_analysis()
    
    print("done!")

if __name__ == "__main__":
    main() 