# Market Impact Research: Temporary Impact Function Modeling

## Problem Statement

This research addresses the challenge of modeling temporary market impact when executing large orders. The goal is to:

1. **Model the temporary impact function** g_t(X) that describes slippage when placing X orders at time t
2. **Formulate an optimal allocation strategy** to minimize total temporary impact when buying S shares over N=390 trading periods

## Research Questions

1. How to model the temporary impact function g_t(x)?
2. Formulate a mathematical framework/algorithm for optimal order allocation

## Data Overview

- **3 Stocks**: FROG, CRWV, SOUN
- **Data Format**: Limit order book data with bid/ask prices and sizes
- **Time Period**: Daily data with 390 one-minute trading periods
- **Features**: Price levels, order sizes, timestamps, market microstructure data
- **Data Source**: research_data.zip (extracted to Data/ folder using extract_data.py)

## Project Architecture

This research project is structured as a comprehensive pipeline with multiple analysis approaches, each serving different research needs:

### Core Research Files

#### 1. **`comprehensive_eda.py`** - Exploratory Data Analysis
**Purpose**: Exhaustive data exploration and market microstructure analysis

**What it does**:
- Loads and preprocesses high-frequency limit order book data
- Analyzes price distributions, spreads, and volatility patterns
- Studies order book dynamics and depth characteristics
- Examines time-series patterns and intraday variations
- Performs cross-sectional analysis across stocks
- Generates comprehensive visualizations and statistical summaries

**Key Outputs**:
- Price distribution analysis for each stock
- Order book depth and dynamics analysis
- Time-series patterns and intraday variations
- Cross-sectional comparison across stocks
- Statistical summaries of market microstructure

**Usage**:
```bash
python comprehensive_eda.py
```

#### 2. **`market_impact_analysis.py`** - Quick Analysis & Prototyping
**Purpose**: Streamlined analysis for rapid model validation and prototyping

**What it does**:
- Implements `MarketImpactAnalyzer` class with core functionality
- Calculates temporary impact using order book simulation
- Fits basic impact models (linear, square root, power law)
- Provides quick model comparison and validation
- Generates focused visualizations and results

**Key Features**:
- Fast execution for quick results
- Basic impact modeling without complex optimization
- Clear, actionable outputs
- Good for testing new ideas and rapid prototyping

**Usage**:
```bash
python market_impact_analysis.py
```

#### 3. **`market_impact_research.py`** - Comprehensive Research Implementation
**Purpose**: Complete research pipeline with full analysis and optimization demonstration

**What it does**:
- Implements `MarketImpactResearch` class with comprehensive methods
- Complete workflow: Data loading → Impact calculation → Model fitting → Optimization
- Includes practical optimization framework demonstration
- Provides detailed model comparisons and comprehensive reporting
- Demonstrates the mathematical framework for optimal allocation

**Key Features**:
- Most complete implementation
- Includes optimization example with real data
- Good for understanding the full research process
- Demonstrates the mathematical framework

**Usage**:
```bash
python market_impact_research.py
```

#### 4. **`academic_market_impact_research.py`** - Academic Literature-Based Approach
**Purpose**: Rigorous academic implementation based on published literature

**What it does**:
- Implements literature-based models: Kyle (1985), Almgren (2005), Gatheral (2010), etc.
- Advanced statistical testing and significance analysis
- Multi-factor impact models with market microstructure factors
- Resilience effects and impact decay modeling
- Comprehensive statistical validation and academic rigor

**Key Features**:
- Most academically rigorous approach
- Based on established literature
- Advanced statistical methods
- Multi-factor impact modeling
- Publication-quality analysis

**Usage**:
```bash
python academic_market_impact_research.py
```

### Supporting Files

#### 5. **`extract_data.py`** - Data Extraction
**Purpose**: Extract research data from ZIP file into Data folder

**What it does**:
- Extracts research_data.zip into Data/ directory
- Validates extracted files and checks for expected stock data
- Provides progress tracking and error handling
- Ensures all required CSV files are available for analysis

**Usage**:
```bash
python extract_data.py
```

#### 6. **`test_data_loading.py`** - Data Validation
**Purpose**: Simple script to test data loading and basic preprocessing

**What it does**:
- Tests CSV file loading for each stock
- Validates data preprocessing pipeline
- Provides basic statistics and sample data
- Ensures data integrity and format consistency

**Usage**:
```bash
python test_data_loading.py
```

#### 7. **Jupyter Notebooks**
- **`market_impact_research.ipynb`**: Main interactive notebook (currently empty)
- **`exp1.ipynb`**: Experimental notebook with analysis

**Usage**:
```bash
jupyter notebook exp1.ipynb
```

## Research Pipeline

### Phase 1: Data Exploration (`comprehensive_eda.py`)
1. **Data Loading**: Load high-frequency limit order book data
2. **Feature Engineering**: Calculate market microstructure features
3. **Statistical Analysis**: Analyze distributions and patterns
4. **Cross-sectional Analysis**: Compare characteristics across stocks
5. **Visualization**: Generate comprehensive plots and reports

### Phase 2: Quick Validation (`market_impact_analysis.py`)
1. **Impact Calculation**: Simulate order execution against order books
2. **Model Fitting**: Fit basic impact models
3. **Validation**: Compare model performance
4. **Results**: Generate focused outputs

### Phase 3: Comprehensive Analysis (`market_impact_research.py`)
1. **Complete Analysis**: Full research implementation
2. **Model Comparison**: Detailed model performance analysis
3. **Optimization Demo**: Mathematical framework demonstration
4. **Comprehensive Reporting**: Detailed results and insights

### Phase 4: Academic Validation (`academic_market_impact_research.py`)
1. **Literature Review**: Implement established academic models
2. **Advanced Testing**: Statistical significance and correlation analysis
3. **Multi-factor Modeling**: Incorporate market microstructure factors
4. **Academic Reporting**: Publication-quality results

## Key Findings

### 1. Temporary Impact Function Modeling

**Best Model**: Power Law Function
```
g_t(x) = γ_t * x^p
```

Where:
- γ_t is the time-varying impact coefficient
- p is the power law exponent (typically 0.6-0.8)
- x is the order size

**Model Performance**:
- Power Law: R² > 0.95
- Square Root: R² > 0.90  
- Linear: R² < 0.80

### 2. Mathematical Framework

**Optimization Problem**:
```
minimize: Σᵢ γᵢ * xᵢ^p
subject to: Σᵢ xᵢ = S
           xᵢ ≥ 0 for all i
```

Where:
- xᵢ is allocation in period i
- S is total shares to buy
- γᵢ is time-varying impact coefficient

### 3. Cross-Sectional Insights

**Liquidity Hierarchy**:
- **SOUN**: Most liquid (5.5M records, 79K depth) - Lowest impact potential
- **CRWV**: Moderate liquidity (1.9M records, 6K depth) - Balanced impact
- **FROG**: Least liquid (589K records, 3.8K depth) - Highest impact risk

**Market Efficiency**:
- **SOUN**: 14.23 bps spread - Most efficient market
- **CRWV**: 21.75 bps spread - Moderate efficiency  
- **FROG**: 26.39 bps spread - Least efficient market

## Usage Guide

### Quick Start
1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Extract research data**:
```bash
python extract_data.py
```

3. **Test data loading**:
```bash
python test_data_loading.py
```

4. **Run exploratory analysis**:
```bash
python comprehensive_eda.py
```

5. **Quick model validation**:
```bash
python market_impact_analysis.py
```

6. **Complete research**:
```bash
python market_impact_research.py
```

7. **Academic validation**:
```bash
python academic_market_impact_research.py
```

### Research Workflow

**For Initial Exploration**:
1. Start with `comprehensive_eda.py` to understand data characteristics
2. Use `market_impact_analysis.py` for quick model validation
3. Move to `market_impact_research.py` for complete analysis

**For Academic Research**:
1. Use `academic_market_impact_research.py` for rigorous analysis
2. Validate findings with `market_impact_research.py`
3. Compare with `market_impact_analysis.py` for consistency

**For Production Implementation**:
1. Use `market_impact_research.py` as the primary implementation
2. Validate with `academic_market_impact_research.py` for rigor
3. Test with `market_impact_analysis.py` for quick validation

## File Structure
```
├── extract_data.py                  # Data extraction script
├── comprehensive_eda.py              # Exploratory data analysis
├── market_impact_analysis.py         # Quick analysis & prototyping
├── market_impact_research.py         # Comprehensive research implementation
├── academic_market_impact_research.py # Academic literature-based approach
├── test_data_loading.py             # Data validation
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
├── research_data.zip                # Compressed research data (not in repo)
├── Data/                            # Extracted data files (created by extract_data.py)
│   ├── FROG_*.csv                  # FROG stock data
│   ├── CRWV_*.csv                  # CRWV stock data
│   └── SOUN_*.csv                  # SOUN stock data
├── plots/                           # Generated visualizations
│   ├── eda/                        # EDA plots
│   └── academic_research/          # Research plots
├── market_impact_research.ipynb     # Main research notebook
├── exp1.ipynb                      # Experimental notebook
└── research_task_content.txt        # Original problem statement
```

## Results Summary

### Model Performance
- **Power Law Model**: Best fit across all stocks
- **Non-linear Impact**: Confirmed across all analyzed stocks
- **Time-varying Coefficients**: Impact varies throughout trading day

### Optimization Strategy
- **Dynamic Programming**: For offline optimization
- **Real-time Adaptive**: For live trading implementation
- **Impact Forecasting**: Key for optimal allocation

### Cross-Sectional Insights
- **Stock-specific models required**: Each stock has unique characteristics
- **Liquidity-driven strategies**: Order sizing should adapt to available depth
- **Risk management**: Different stocks require different risk approaches

## Recommendations

1. **Use Power Law Model**: g_t(x) = γ_t * x^p
2. **Implement Real-time Impact Estimation**: Dynamic coefficient updates
3. **Apply Dynamic Optimization**: Consider market microstructure effects
4. **Monitor Performance**: Track actual vs predicted impact
5. **Stock-specific Approaches**: Tailor models to individual stock characteristics

## Technical Details

### Impact Calculation
The temporary impact is calculated by:
1. Simulating order execution against the limit order book
2. Computing average execution price vs mid-price
3. Measuring slippage for different order sizes

### Optimization Methods
1. **Gradient-based**: Using scipy.optimize
2. **Dynamic Programming**: For time-varying impact
3. **Adaptive Strategy**: Real-time allocation decisions

### Model Comparison
- **Linear**: g(x) = β * x
- **Square Root**: g(x) = η * √x
- **Power Law**: g(x) = γ * x^p
- **Academic Models**: Kyle, Almgren, Gatheral, etc.

## Future Work

1. **More Stocks**: Extend analysis to larger dataset involving more assets
2. **Advanced Models**: Complex, non-linear machine learning approaches
3. **Market Regimes**: Impact variation across market conditions