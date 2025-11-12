# IBD Style Stock Screener

A comprehensive stock screener implementing Investor's Business Daily (IBD) style screening criteria with multiprocessing for efficient data fetching.

## Features

This screener implements 6 IBD-style screening criteria:

### 1. Momentum 97
- **Criteria**: 1M/3M/6M Rank (Pct) ≥ 97%
- **Purpose**: Identifies stocks with exceptional momentum across multiple timeframes

### 2. Explosive Estimated EPS Growth Stocks
- **Criteria**:
  - RS Rating ≥ 80
  - EPS Est Cur Qtr % ≥ 100%
  - 50-Day Avg Vol (1000s) ≥ 100
  - Price vs 50-Day ≥ 0.0%
- **Purpose**: Finds stocks with strong relative strength and explosive earnings growth estimates

### 3. Up on Volume List
- **Criteria**:
  - Price % Chg ≥ 0.00%
  - Vol% Chg vs 50-Day ≥ 20%
  - Current Price ≥ $10
  - 50-Day Avg Vol (1000s) ≥ 100
  - Market Cap (mil) ≥ $250
  - RS Rating ≥ 80
  - EPS % Chg Last Qtr ≥ 20%
  - A/D Rating ABC
- **Purpose**: Identifies stocks moving up on increased volume with institutional buying

### 4. Top 2% RS Rating List
- **Criteria**:
  - RS Rating ≥ 98
  - 10Day > 21Day > 50Day (ascending moving averages)
  - 50-Day Avg Vol (1000s) ≥ 100
  - Volume (1000s) ≥ 100
  - Sector NOT: medical
- **Purpose**: Finds the strongest stocks with proper moving average alignment

### 5. 4% Bullish Yesterday
- **Criteria**:
  - Price ≥ $1
  - Change > 4%
  - Market cap > $250M
  - Volume > 100K
  - Rel Volume > 1
  - Change from Open > 0%
  - Avg Volume 90D > 100K
- **Purpose**: Captures strong daily movers with good liquidity

### 6. Healthy Chart Watch List
- **Criteria**:
  - 10Day > 21Day > 50Day
  - 50Day > 150Day > 200Day
  - RS Line New High
  - RS Rating ≥ 90
  - A/D Rating AB
  - Ind Group RS AB (not fully implemented - proprietary)
  - Comp Rating ≥ 80
  - 50-Day Avg Vol (1000s) ≥ 100
- **Purpose**: Identifies stocks in healthy uptrends with institutional support

## IBD Indicators Implemented

### RS Rating (Relative Strength Rating)
- **Formula**: `0.4 × ROC(63d) + 0.2 × ROC(126d) + 0.2 × ROC(189d) + 0.2 × ROC(252d)`
- Percentile ranked against all stocks (1-99 scale)
- Weights recent 3-month performance at 40%, older quarters at 20% each

### RS Line
- **Formula**: `Stock Price / S&P 500 Price`
- Tracks relative performance vs market benchmark
- RS Line New High indicates outperformance at new levels

### A/D Rating (Accumulation/Distribution Rating)
- Based on 13 weeks (65 trading days) of price/volume data
- Analyzes money flow and up/down volume ratios
- Rating scale: A+ to E

### Composite Rating
- Combines RS Rating (40%), EPS Rating (40%), A/D Rating (20%)
- Comprehensive measure of stock quality
- Scale: 1-99

### Moving Averages
- Calculates 10, 21, 50, 150, 200-day moving averages
- Checks for proper alignment (Stage 2 uptrend)

### Volume Metrics
- 50-day and 90-day average volume
- Relative volume (current vs average)
- Volume change percentage

## Setup

### Prerequisites

1. **Python 3.8+** required
2. **FMP API Key** (Premium Plan recommended for 750 calls/min)
   - Get your API key from: https://financialmodelingprep.com/developer/docs/

### Installation

1. Clone the repository:
```bash
cd MarketAlgo
```

2. Install dependencies:
```bash
pip install pandas numpy python-dotenv curl-cffi
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your FMP_API_KEY
```

### Environment Variables

Edit `.env` file:

```bash
# Required
FMP_API_KEY=your_fmp_api_key_here

# Optional (defaults shown)
FMP_RATE_LIMIT=750              # API calls per minute (Premium Plan)
ORATNEK_MAX_WORKERS=10          # Multiprocessing workers
```

## Usage

### Basic Usage

Run the screener:

```bash
python ibd_screener.py
```

This will:
1. Fetch all NASDAQ and NYSE tickers (excluding ETFs/funds)
2. Download S&P 500 data for RS Line calculations
3. Analyze all stocks with multiprocessing
4. Apply all 6 screening criteria
5. Save results to `screener_results/` directory

### Output Files

The screener creates the following files in `screener_results/`:

- `all_stocks_analyzed.csv` - Complete dataset with all calculated metrics
- `momentum_97.csv` - Stocks passing Momentum 97 screen
- `explosive_eps_growth.csv` - Explosive EPS growth stocks
- `up_on_volume.csv` - Up on volume list
- `top_2pct_rs.csv` - Top 2% RS rating stocks
- `4pct_bullish.csv` - 4% bullish yesterday stocks
- `healthy_chart.csv` - Healthy chart watch list
- `summary.csv` - Summary report with top stocks per screen

### Custom Usage

```python
from ibd_screener import IBDScreener

# Initialize screener
screener = IBDScreener(
    api_key="your_api_key",  # Optional if in .env
    max_workers=10           # Adjust for your system
)

# Run screener
screens = screener.run_screener(output_dir="my_results")

# Access individual screens
momentum_stocks = screens['momentum_97']
explosive_eps = screens['explosive_eps_growth']
```

## Performance

### Multiprocessing
- Uses `ProcessPoolExecutor` for parallel data fetching
- Default: 10 workers (configurable via `ORATNEK_MAX_WORKERS`)
- Respects FMP API rate limits (750 calls/min for Premium)

### Expected Runtime
- ~6,000 stocks (NASDAQ + NYSE)
- 10 workers
- Premium API (750 calls/min)
- **Estimated time: 20-30 minutes**

### Optimization Tips

1. **Increase workers** if you have a powerful CPU:
   ```bash
   ORATNEK_MAX_WORKERS=20
   ```

2. **Upgrade API plan** for higher rate limits:
   - Professional Plan: 1500 calls/min
   - Enterprise Plan: Custom limits

3. **Filter tickers** before analysis:
   ```python
   # Only analyze specific sectors
   tickers = screener.get_tickers()
   tech_tickers = [t for t in tickers if is_tech(t)]
   ```

## Data Columns

### Price Metrics
- `symbol`, `company_name`, `sector`
- `price`, `price_change`, `price_change_pct`
- `change_from_open_pct`
- `price_vs_50day` - % above/below 50-day MA

### Volume Metrics
- `volume` - Current volume (thousands)
- `avg_volume_50` - 50-day average volume
- `avg_volume_90` - 90-day average volume
- `volume_vs_50day_pct` - Volume change vs 50-day average
- `rel_volume` - Relative volume ratio

### IBD Indicators
- `rs_rating` - Relative Strength Rating (1-99)
- `rs_strength` - Raw RS strength factor
- `rs_line_new_high` - Boolean, RS line at new high
- `ad_rating` - Accumulation/Distribution Rating (A+ to E)
- `composite_rating` - Composite Rating (1-99)

### Momentum Metrics
- `momentum_1m` - 1-month return %
- `momentum_3m` - 3-month return %
- `momentum_6m` - 6-month return %
- `momentum_1m_rank_pct` - 1-month percentile rank
- `momentum_3m_rank_pct` - 3-month percentile rank
- `momentum_6m_rank_pct` - 6-month percentile rank

### EPS Metrics
- `eps_growth_last_qtr` - EPS growth last quarter %
- `eps_est_cur_qtr_pct` - EPS estimate current quarter %
- `eps_rating` - EPS Rating (1-99)

### Moving Averages
- `ma10`, `ma21`, `ma50`, `ma150`, `ma200`

### Fundamentals
- `market_cap_mil` - Market capitalization (millions)

## Technical Details

### RS Rating Calculation

The RS Rating is calculated using IBD's documented formula:

```python
# Calculate rate of change for different periods
ROC_63 = (Close_today / Close_63_days_ago) - 1
ROC_126 = (Close_today / Close_126_days_ago) - 1
ROC_189 = (Close_today / Close_189_days_ago) - 1
ROC_252 = (Close_today / Close_252_days_ago) - 1

# IBD formula: 40% weight on recent 3 months
RS_Strength = 0.4 * ROC_63 + 0.2 * ROC_126 + 0.2 * ROC_189 + 0.2 * ROC_252

# Convert to percentile rank (1-99)
RS_Rating = percentile_rank(RS_Strength) * 100
```

### A/D Rating Calculation

Approximation based on publicly available information:

```python
# Money Flow Multiplier
MF_Multiplier = ((Close - Low) - (High - Close)) / (High - Low)

# Money Flow Volume
MF_Volume = MF_Multiplier * Volume

# Calculate over 13 weeks (65 days)
AD_Score = sum(MF_Volume) / sum(Volume)

# Also consider up/down volume ratio
Up_Volume_Ratio = Up_Days_Volume / Down_Days_Volume

# Combined score mapped to letter grades (A+ to E)
```

### RS Line New High

```python
RS_Line = Stock_Price / SP500_Price
RS_Line_New_High = (Current_RS_Line >= Max_RS_Line_252d * 0.99)
```

## Limitations & Notes

### Proprietary Metrics
Some IBD metrics are proprietary and not fully disclosed:
- **Exact A/D Rating formula** - Our implementation is an approximation
- **Industry Group RS** - Not implemented (requires industry classification)
- **EPS Rating** - Simplified (requires historical earnings database)
- **Composite Rating weights** - Approximated based on public sources

### API Limitations
- **Rate limits**: Respect FMP API limits (750 calls/min for Premium)
- **Data availability**: Some stocks may have incomplete data
- **Historical data**: Requires 252+ days of data for full RS Rating

### Known Issues
- **EPS estimates**: Current quarter estimates may not be available for all stocks
- **Sector filtering**: Medical sector filter is case-sensitive
- **Industry Group RS**: Not implemented due to lack of IBD industry classification

## Troubleshooting

### Rate Limit Errors
If you see rate limit errors:
```bash
# Reduce workers
ORATNEK_MAX_WORKERS=5

# Or verify your API plan rate limit
FMP_RATE_LIMIT=750  # Adjust based on your plan
```

### Memory Issues
For systems with limited RAM:
```bash
# Reduce workers
ORATNEK_MAX_WORKERS=5

# Or process in batches
python ibd_screener.py --batch-size 1000
```

### Incomplete Data
Some stocks may fail analysis due to:
- Newly listed (insufficient historical data)
- Delisted or suspended trading
- API data unavailable

These are automatically skipped with a warning message.

## References

### IBD Methodology
- [IBD RS Rating Documentation](https://www.investors.com/ibd-university/find-evaluate-stocks/exclusive-ratings/)
- [William O'Neil + Co. Proprietary Ratings](https://www.williamoneil.com/proprietary-ratings-and-rankings/)

### FinancialModelingPrep API
- [FMP API Documentation](https://financialmodelingprep.com/developer/docs/)
- [Stock Screener API](https://financialmodelingprep.com/developer/docs/stock-screener-api)

## License

See repository license.

## Contributing

Contributions welcome! Please ensure:
1. IBD methodology accuracy
2. Proper error handling
3. Rate limit compliance
4. Documentation updates

## Support

For issues or questions:
1. Check troubleshooting section
2. Review FMP API documentation
3. Open GitHub issue with detailed description

---

**Disclaimer**: This screener is for educational and research purposes. Not financial advice. Always do your own due diligence before making investment decisions.
