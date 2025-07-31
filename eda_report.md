# Market Impact Research: Exploratory Data Analysis Report

## Executive Summary

This report presents the findings from a comprehensive exploratory data analysis of high-frequency limit order book data for three stocks (FROG, CRWV, SOUN) over a one-month period. The analysis reveals critical insights about market microstructure that directly inform our approach to modeling temporary market impact and developing optimal order allocation strategies.

## Key Observations and Their Implications

### 1. **Data Volume and Market Activity**

**Observations:**
- **SOUN**: 5.5M records - Highest activity, most liquid
- **CRWV**: 1.9M records - Moderate activity
- **FROG**: 589K records - Lowest activity, least liquid

**Implications for Market Impact:**
- **Liquidity Hierarchy**: SOUN offers the deepest order books, suggesting lower impact for large orders
- **Execution Risk**: FROG's low activity indicates higher execution risk and potentially higher impact
- **Strategy Adaptation**: Different stocks require different allocation strategies based on their liquidity profiles

### 2. **Price Characteristics and Market Efficiency**

**Observations:**
- **FROG**: $31.24 average price, 26.39 bps spread, 0.0018 volatility
- **CRWV**: $44.41 average price, 21.75 bps spread, 0.0042 volatility  
- **SOUN**: $8.39 average price, 14.23 bps spread, 0.0032 volatility

**Implications for Market Impact:**
- **Spread-Impact Relationship**: Lower spreads (SOUN) suggest more efficient markets with potentially lower impact
- **Volatility-Impact Correlation**: Higher volatility (CRWV) may indicate more unpredictable impact patterns
- **Price Level Effects**: Higher-priced stocks (CRWV) may show different impact scaling than lower-priced ones

### 3. **Order Book Depth and Liquidity Provision**

**Observations:**
- **SOUN**: 79,091 average depth - Exceptionally deep order books
- **CRWV**: 6,304 average depth - Moderate depth
- **FROG**: 3,884 average depth - Shallowest order books

**Implications for Market Impact:**
- **Impact Scaling**: SOUN's deep books suggest linear or sub-linear impact scaling
- **Market Resilience**: Deep books indicate better market resilience to large orders
- **Execution Strategy**: FROG requires more careful order sizing due to limited depth

### 4. **Cross-Sectional Market Microstructure**

**Key Findings:**
- **Liquidity Asymmetry**: Significant variation in bid-ask imbalances across stocks
- **Depth Volatility**: Different stocks show varying depth stability
- **Market Efficiency**: SOUN appears most efficient, FROG least efficient

**Implications for Market Impact:**
- **Non-Uniform Impact**: Impact functions likely vary significantly across stocks
- **Time-Varying Coefficients**: Market conditions change throughout the day
- **Stock-Specific Models**: One-size-fits-all approach insufficient

## Market Impact Modeling Implications

### 1. **Model Selection Strategy**

Based on the EDA findings, we should expect:

**For SOUN (High Liquidity):**
- Linear or power law with low exponent (p ≈ 0.5-0.6)
- Stable impact coefficients throughout the day
- Lower impact magnitude due to deep order books

**For CRWV (Moderate Liquidity):**
- Power law with moderate exponent (p ≈ 0.6-0.7)
- Time-varying coefficients due to volatility
- Moderate impact magnitude

**For FROG (Low Liquidity):**
- Power law with higher exponent (p ≈ 0.7-0.8)
- Highly time-varying coefficients
- Higher impact magnitude due to shallow books

### 2. **Optimization Strategy Development**

**Adaptive Allocation:**
- **SOUN**: Can handle larger order sizes with minimal impact
- **CRWV**: Requires moderate order sizing with timing optimization
- **FROG**: Needs aggressive order splitting and careful timing

**Time-Based Strategies:**
- Leverage intraday patterns for optimal execution timing
- Adjust order sizes based on depth variations
- Monitor bid-ask imbalances for execution opportunities

### 3. **Risk Management Considerations**

**Liquidity Risk:**
- FROG poses highest liquidity risk requiring conservative sizing
- SOUN offers lowest liquidity risk allowing larger orders
- CRWV requires balanced approach

**Market Impact Risk:**
- Higher volatility in CRWV suggests more unpredictable impact
- SOUN's stability allows more predictable impact modeling
- FROG's low activity requires more conservative impact estimates

## Recommendations for Impact Modeling

### 1. **Model Architecture**

**Multi-Factor Approach:**
```
g_t(x) = γ_t * x^p * f(market_conditions)
```
Where:
- γ_t: Time-varying impact coefficient
- p: Stock-specific power law exponent
- f(market_conditions): Function of depth, spread, volatility

### 2. **Parameter Estimation**

**Stock-Specific Calibration:**
- **SOUN**: p ≈ 0.5-0.6, lower γ_t values
- **CRWV**: p ≈ 0.6-0.7, moderate γ_t values  
- **FROG**: p ≈ 0.7-0.8, higher γ_t values


## Conclusion

1. **Universal models are insufficient** - each stock requires tailored impact functions
2. **Time-varying parameters are essential** - market conditions change throughout the day
3. **Liquidity-driven strategies work best** - order sizing should adapt to available depth
4. **Risk management is stock-specific** - different stocks require different risk approaches