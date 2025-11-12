"""
IBD Style Stock Screener with Multiprocessing

This screener implements the following IBD-style screening criteria:
1. Momentum 97
2. Explosive Estimated EPS Growth Stocks
3. Up on Volume List
4. Top 2% RS Rating List
5. 4% Bullish Yesterday
6. Healthy Chart Watch List

Uses FinancialModelingPrep API and multiprocessing for efficient data fetching.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from curl_cffi.requests import Session
from dotenv import load_dotenv
import multiprocessing as mp
from functools import partial
import warnings

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Import get_tickers module
from get_tickers import FMPTickerFetcher


class RateLimiter:
    """Thread-safe rate limiter for API calls"""

    def __init__(self, rate_limit: int = 750):
        self.rate_limit = rate_limit
        self.request_timestamps = []
        self.lock = mp.Lock()

    def wait_if_needed(self):
        """Wait if rate limit is reached"""
        with self.lock:
            current_time = time.time()
            # Remove timestamps older than 60 seconds
            self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]

            if len(self.request_timestamps) >= self.rate_limit:
                # Sleep until the oldest request is older than 60 seconds
                sleep_time = 60 - (current_time - self.request_timestamps[0]) + 0.1
                time.sleep(sleep_time)
                # Trim the list again after sleeping
                current_time = time.time()
                self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]

            self.request_timestamps.append(current_time)


class FMPDataFetcher:
    """Fetch stock data from FinancialModelingPrep API"""

    def __init__(self, api_key: str, rate_limiter: RateLimiter = None):
        self.api_key = api_key
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = Session(impersonate="chrome110")
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with rate limiting and error handling"""
        self.rate_limiter.wait_if_needed()

        if params is None:
            params = {}
        params['apikey'] = self.api_key

        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching {endpoint}: {e}")
            return None

    def get_historical_prices(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Get historical price data"""
        data = self._make_request(f"historical-price-full/{symbol}", {"timeseries": days})

        if not data or 'historical' not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote"""
        data = self._make_request(f"quote/{symbol}")
        return data[0] if data and len(data) > 0 else {}

    def get_key_metrics(self, symbol: str) -> Dict:
        """Get key metrics"""
        data = self._make_request(f"key-metrics/{symbol}", {"limit": 1})
        return data[0] if data and len(data) > 0 else {}

    def get_financial_growth(self, symbol: str) -> Dict:
        """Get financial growth data"""
        data = self._make_request(f"financial-growth/{symbol}", {"limit": 4})
        return data if data else []

    def get_analyst_estimates(self, symbol: str) -> Dict:
        """Get analyst estimates"""
        data = self._make_request(f"analyst-estimates/{symbol}", {"limit": 4})
        return data if data else []

    def get_sp500_data(self, days: int = 365) -> pd.DataFrame:
        """Get S&P 500 (SPY) historical data"""
        return self.get_historical_prices("SPY", days)


class IBDIndicators:
    """Calculate IBD-style indicators"""

    @staticmethod
    def calculate_rs_rating(price_data: pd.DataFrame) -> float:
        """
        Calculate IBD RS Rating
        Formula: 0.4 * ROC(63) + 0.2 * ROC(126) + 0.2 * ROC(189) + 0.2 * ROC(252)

        Returns a strength factor that will be converted to percentile rank later
        """
        if len(price_data) < 252:
            return 0.0

        closes = price_data['close'].values

        # Calculate Rate of Change for different periods
        roc_63 = (closes[-1] / closes[-63] - 1) if len(closes) >= 63 else 0
        roc_126 = (closes[-1] / closes[-126] - 1) if len(closes) >= 126 else 0
        roc_189 = (closes[-1] / closes[-189] - 1) if len(closes) >= 189 else 0
        roc_252 = (closes[-1] / closes[-252] - 1) if len(closes) >= 252 else 0

        # IBD formula with 40% weight on recent 3 months
        rs_strength = 0.4 * roc_63 + 0.2 * roc_126 + 0.2 * roc_189 + 0.2 * roc_252

        return rs_strength

    @staticmethod
    def calculate_momentum_rank(price_data: pd.DataFrame, period_days: int) -> float:
        """Calculate momentum rank for specific period (1M, 3M, 6M)"""
        if len(price_data) < period_days:
            return 0.0

        closes = price_data['close'].values
        return (closes[-1] / closes[-period_days] - 1) * 100

    @staticmethod
    def calculate_rs_line(price_data: pd.DataFrame, sp500_data: pd.DataFrame) -> pd.Series:
        """
        Calculate RS Line (Stock Price / S&P 500 Price)
        """
        # Merge on date
        merged = pd.merge(
            price_data[['date', 'close']],
            sp500_data[['date', 'close']],
            on='date',
            suffixes=('_stock', '_sp500')
        )

        if len(merged) == 0:
            return pd.Series()

        merged['rs_line'] = merged['close_stock'] / merged['close_sp500']
        return merged['rs_line']

    @staticmethod
    def is_rs_line_new_high(rs_line: pd.Series, lookback: int = 252) -> bool:
        """Check if RS line is at new high"""
        if len(rs_line) < 2:
            return False

        recent_high = rs_line.iloc[-lookback:].max() if len(rs_line) >= lookback else rs_line.max()
        current = rs_line.iloc[-1]

        return current >= recent_high * 0.99  # Within 1% of high

    @staticmethod
    def calculate_ad_rating(price_data: pd.DataFrame) -> str:
        """
        Calculate Accumulation/Distribution Rating (approximation)
        Based on 13 weeks (65 trading days) of price and volume data

        Returns rating: A+, A, A-, B+, B, B-, C+, C, C-, D+, D, D-, E
        """
        if len(price_data) < 65:
            return "C"

        recent_data = price_data.tail(65).copy()

        # Calculate Money Flow Multiplier
        recent_data['mf_multiplier'] = (
            (recent_data['close'] - recent_data['low']) -
            (recent_data['high'] - recent_data['close'])
        ) / (recent_data['high'] - recent_data['low'])
        recent_data['mf_multiplier'].fillna(0, inplace=True)

        # Money Flow Volume
        recent_data['mf_volume'] = recent_data['mf_multiplier'] * recent_data['volume']

        # Calculate accumulation/distribution score
        ad_score = recent_data['mf_volume'].sum() / recent_data['volume'].sum()

        # Also consider up/down volume ratio
        up_days = recent_data[recent_data['close'] > recent_data['close'].shift(1)]
        down_days = recent_data[recent_data['close'] < recent_data['close'].shift(1)]

        up_volume = up_days['volume'].sum()
        down_volume = down_days['volume'].sum()

        volume_ratio = up_volume / (down_volume + 1)  # Avoid division by zero

        # Combined score
        combined_score = (ad_score + (volume_ratio - 1)) / 2

        # Map to letter grades
        if combined_score > 0.4:
            return "A+"
        elif combined_score > 0.3:
            return "A"
        elif combined_score > 0.2:
            return "A-"
        elif combined_score > 0.1:
            return "B+"
        elif combined_score > 0.05:
            return "B"
        elif combined_score > 0:
            return "B-"
        elif combined_score > -0.05:
            return "C+"
        elif combined_score > -0.1:
            return "C"
        elif combined_score > -0.15:
            return "C-"
        elif combined_score > -0.2:
            return "D+"
        elif combined_score > -0.3:
            return "D"
        elif combined_score > -0.4:
            return "D-"
        else:
            return "E"

    @staticmethod
    def calculate_moving_averages(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate moving averages"""
        mas = {}

        for period in [10, 21, 50, 150, 200]:
            if len(price_data) >= period:
                mas[f'ma{period}'] = price_data['close'].tail(period).mean()
            else:
                mas[f'ma{period}'] = None

        return mas

    @staticmethod
    def calculate_volume_metrics(price_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-related metrics"""
        if len(price_data) < 50:
            return {
                'avg_volume_50': 0,
                'avg_volume_90': 0,
                'current_volume': 0,
                'volume_vs_50day_pct': 0,
                'rel_volume': 0
            }

        current_volume = price_data['volume'].iloc[-1]
        avg_volume_50 = price_data['volume'].tail(50).mean()
        avg_volume_90 = price_data['volume'].tail(min(90, len(price_data))).mean()

        return {
            'avg_volume_50': avg_volume_50 / 1000,  # In thousands
            'avg_volume_90': avg_volume_90 / 1000,
            'current_volume': current_volume / 1000,
            'volume_vs_50day_pct': ((current_volume / avg_volume_50 - 1) * 100) if avg_volume_50 > 0 else 0,
            'rel_volume': current_volume / avg_volume_50 if avg_volume_50 > 0 else 0
        }

    @staticmethod
    def calculate_eps_metrics(growth_data: List[Dict], estimates_data: List[Dict]) -> Dict[str, float]:
        """Calculate EPS-related metrics"""
        eps_metrics = {
            'eps_growth_last_qtr': 0,
            'eps_est_cur_qtr_pct': 0,
            'eps_rating': 50
        }

        # EPS growth last quarter
        if growth_data and len(growth_data) >= 2:
            try:
                latest = growth_data[0]
                if 'epsgrowth' in latest and latest['epsgrowth'] is not None:
                    eps_metrics['eps_growth_last_qtr'] = latest['epsgrowth'] * 100
            except:
                pass

        # EPS estimate current quarter
        if estimates_data and len(estimates_data) > 0:
            try:
                latest_estimate = estimates_data[0]
                if 'estimatedEpsAvg' in latest_estimate and latest_estimate['estimatedEpsAvg'] is not None:
                    # Calculate growth vs last year same quarter
                    eps_metrics['eps_est_cur_qtr_pct'] = 0  # Simplified, would need historical data
            except:
                pass

        return eps_metrics

    @staticmethod
    def calculate_composite_rating(rs_rating: float, eps_rating: float, ad_rating: str) -> float:
        """
        Calculate Composite Rating (approximation)
        Combines RS Rating, EPS Rating, A/D Rating
        Weights: RS (40%), EPS (40%), A/D (20%)
        """
        # Convert A/D rating to numeric
        ad_map = {
            'A+': 95, 'A': 90, 'A-': 85,
            'B+': 80, 'B': 75, 'B-': 70,
            'C+': 65, 'C': 60, 'C-': 55,
            'D+': 50, 'D': 45, 'D-': 40,
            'E': 30
        }
        ad_numeric = ad_map.get(ad_rating, 60)

        composite = 0.4 * rs_rating + 0.4 * eps_rating + 0.2 * ad_numeric
        return min(99, max(1, composite))


class StockAnalyzer:
    """Analyze individual stocks"""

    def __init__(self, api_key: str, sp500_data: pd.DataFrame):
        self.api_key = api_key
        self.sp500_data = sp500_data
        self.fetcher = FMPDataFetcher(api_key)

    def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """Analyze a single stock and return all calculated metrics"""
        try:
            # Fetch all required data
            price_data = self.fetcher.get_historical_prices(symbol, days=365)
            if price_data.empty or len(price_data) < 100:
                return None

            quote = self.fetcher.get_quote(symbol)
            if not quote:
                return None

            metrics = self.fetcher.get_key_metrics(symbol)
            growth_data = self.fetcher.get_financial_growth(symbol)
            estimates_data = self.fetcher.get_analyst_estimates(symbol)

            # Calculate indicators
            indicators = IBDIndicators()

            # RS Rating (strength factor, will be converted to percentile later)
            rs_strength = indicators.calculate_rs_rating(price_data)

            # Momentum ranks
            momentum_1m = indicators.calculate_momentum_rank(price_data, 21)
            momentum_3m = indicators.calculate_momentum_rank(price_data, 63)
            momentum_6m = indicators.calculate_momentum_rank(price_data, 126)

            # RS Line
            rs_line = indicators.calculate_rs_line(price_data, self.sp500_data)
            rs_line_new_high = indicators.is_rs_line_new_high(rs_line) if not rs_line.empty else False

            # A/D Rating
            ad_rating = indicators.calculate_ad_rating(price_data)

            # Moving averages
            mas = indicators.calculate_moving_averages(price_data)

            # Volume metrics
            volume_metrics = indicators.calculate_volume_metrics(price_data)

            # EPS metrics
            eps_metrics = indicators.calculate_eps_metrics(growth_data, estimates_data)

            # Price metrics
            current_price = quote.get('price', 0)
            price_change = quote.get('change', 0)
            price_change_pct = quote.get('changesPercentage', 0)
            open_price = quote.get('open', current_price)
            change_from_open_pct = ((current_price - open_price) / open_price * 100) if open_price > 0 else 0

            # Price vs 50-day MA
            price_vs_50day = 0
            if mas['ma50'] is not None and mas['ma50'] > 0:
                price_vs_50day = ((current_price - mas['ma50']) / mas['ma50'] * 100)

            # Market cap
            market_cap = quote.get('marketCap', 0) / 1_000_000  # In millions

            # Sector
            sector = quote.get('sector', '')

            # Compile all data
            stock_data = {
                'symbol': symbol,
                'company_name': quote.get('name', ''),
                'sector': sector,
                'price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'change_from_open_pct': change_from_open_pct,
                'market_cap_mil': market_cap,
                'volume': volume_metrics['current_volume'],
                'avg_volume_50': volume_metrics['avg_volume_50'],
                'avg_volume_90': volume_metrics['avg_volume_90'],
                'volume_vs_50day_pct': volume_metrics['volume_vs_50day_pct'],
                'rel_volume': volume_metrics['rel_volume'],
                'rs_strength': rs_strength,  # Will be converted to percentile
                'rs_rating': 0,  # Will be calculated after all stocks
                'momentum_1m': momentum_1m,
                'momentum_3m': momentum_3m,
                'momentum_6m': momentum_6m,
                'rs_line_new_high': rs_line_new_high,
                'ad_rating': ad_rating,
                'eps_growth_last_qtr': eps_metrics['eps_growth_last_qtr'],
                'eps_est_cur_qtr_pct': eps_metrics['eps_est_cur_qtr_pct'],
                'eps_rating': eps_metrics['eps_rating'],
                'composite_rating': 0,  # Will be calculated after RS rating
                'ma10': mas['ma10'],
                'ma21': mas['ma21'],
                'ma50': mas['ma50'],
                'ma150': mas['ma150'],
                'ma200': mas['ma200'],
                'price_vs_50day': price_vs_50day
            }

            return stock_data

        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None


def analyze_stock_wrapper(args):
    """Wrapper function for multiprocessing"""
    symbol, api_key, sp500_data_dict = args

    # Convert sp500_data back to DataFrame
    sp500_data = pd.DataFrame(sp500_data_dict)

    analyzer = StockAnalyzer(api_key, sp500_data)
    return analyzer.analyze_stock(symbol)


class IBDScreener:
    """Main screener class"""

    def __init__(self, api_key: str = None, max_workers: int = None):
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required")

        self.max_workers = max_workers or int(os.getenv('ORATNEK_MAX_WORKERS', '10'))
        self.fetcher = FMPDataFetcher(self.api_key)

    def get_tickers(self) -> List[str]:
        """Get list of tickers using get_tickers.py"""
        print("\n" + "="*60)
        print("Fetching ticker list...")
        print("="*60)

        ticker_fetcher = FMPTickerFetcher(self.api_key)
        df = ticker_fetcher.get_all_stocks(['nasdaq', 'nyse'])

        tickers = df['Ticker'].tolist()
        print(f"✓ Retrieved {len(tickers)} tickers")

        return tickers

    def fetch_and_analyze_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch and analyze all stocks with multiprocessing"""
        print("\n" + "="*60)
        print("Fetching S&P 500 data for RS Line calculation...")
        print("="*60)

        sp500_data = self.fetcher.get_sp500_data(days=365)
        if sp500_data.empty:
            raise ValueError("Failed to fetch S&P 500 data")

        print(f"✓ Fetched {len(sp500_data)} days of S&P 500 data")

        # Convert sp500_data to dict for multiprocessing
        sp500_data_dict = sp500_data.to_dict('records')

        print("\n" + "="*60)
        print(f"Analyzing {len(tickers)} stocks with {self.max_workers} workers...")
        print("="*60)

        all_data = []

        # Prepare arguments for multiprocessing
        args_list = [(ticker, self.api_key, sp500_data_dict) for ticker in tickers]

        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(analyze_stock_wrapper, args): args[0] for args in args_list}

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 100 == 0:
                    print(f"Progress: {completed}/{len(tickers)} stocks analyzed ({completed/len(tickers)*100:.1f}%)")

                result = future.result()
                if result:
                    all_data.append(result)

        print(f"\n✓ Successfully analyzed {len(all_data)} stocks")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Calculate RS Rating percentiles
        print("\nCalculating RS Rating percentiles...")
        df['rs_rating'] = df['rs_strength'].rank(pct=True) * 100

        # Calculate momentum rank percentiles
        df['momentum_1m_rank_pct'] = df['momentum_1m'].rank(pct=True) * 100
        df['momentum_3m_rank_pct'] = df['momentum_3m'].rank(pct=True) * 100
        df['momentum_6m_rank_pct'] = df['momentum_6m'].rank(pct=True) * 100

        # Calculate Composite Rating
        print("Calculating Composite Ratings...")
        indicators = IBDIndicators()
        df['composite_rating'] = df.apply(
            lambda row: indicators.calculate_composite_rating(
                row['rs_rating'],
                row['eps_rating'],
                row['ad_rating']
            ),
            axis=1
        )

        return df

    def apply_screening_criteria(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Apply all screening criteria"""
        print("\n" + "="*60)
        print("Applying screening criteria...")
        print("="*60)

        screens = {}

        # 1. Momentum 97
        print("\n1. Momentum 97 Screen")
        momentum_97 = df[
            (df['momentum_1m_rank_pct'] >= 97) &
            (df['momentum_3m_rank_pct'] >= 97) &
            (df['momentum_6m_rank_pct'] >= 97)
        ].copy()
        print(f"   Found {len(momentum_97)} stocks")
        screens['momentum_97'] = momentum_97

        # 2. Explosive Estimated EPS Growth Stocks
        print("\n2. Explosive Estimated EPS Growth Stocks")
        explosive_eps = df[
            (df['rs_rating'] >= 80) &
            (df['eps_est_cur_qtr_pct'] >= 100) &
            (df['avg_volume_50'] >= 100) &
            (df['price_vs_50day'] >= 0.0)
        ].copy()
        print(f"   Found {len(explosive_eps)} stocks")
        screens['explosive_eps_growth'] = explosive_eps

        # 3. Up on Volume List
        print("\n3. Up on Volume List")
        up_on_volume = df[
            (df['price_change_pct'] >= 0.00) &
            (df['volume_vs_50day_pct'] >= 20) &
            (df['price'] >= 10) &
            (df['avg_volume_50'] >= 100) &
            (df['market_cap_mil'] >= 250) &
            (df['rs_rating'] >= 80) &
            (df['eps_growth_last_qtr'] >= 20) &
            (df['ad_rating'].isin(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-']))
        ].copy()
        print(f"   Found {len(up_on_volume)} stocks")
        screens['up_on_volume'] = up_on_volume

        # 4. Top 2% RS Rating List
        print("\n4. Top 2% RS Rating List")
        top_2_pct_rs = df[
            (df['rs_rating'] >= 98) &
            (df['ma10'] > df['ma21']) &
            (df['ma21'] > df['ma50']) &
            (df['avg_volume_50'] >= 100) &
            (df['volume'] >= 100) &
            (~df['sector'].str.contains('medical', case=False, na=False))
        ].copy()
        print(f"   Found {len(top_2_pct_rs)} stocks")
        screens['top_2pct_rs'] = top_2_pct_rs

        # 5. 4% Bullish Yesterday
        print("\n5. 4% Bullish Yesterday")
        bullish_4pct = df[
            (df['price'] >= 1) &
            (df['price_change_pct'] > 4) &
            (df['market_cap_mil'] > 250) &
            (df['volume'] > 100) &
            (df['rel_volume'] > 1) &
            (df['change_from_open_pct'] > 0) &
            (df['avg_volume_90'] > 100)
        ].copy()
        print(f"   Found {len(bullish_4pct)} stocks")
        screens['4pct_bullish'] = bullish_4pct

        # 6. Healthy Chart Watch List
        print("\n6. Healthy Chart Watch List")
        healthy_chart = df[
            (df['ma10'] > df['ma21']) &
            (df['ma21'] > df['ma50']) &
            (df['ma50'] > df['ma150']) &
            (df['ma150'] > df['ma200']) &
            (df['rs_line_new_high'] == True) &
            (df['rs_rating'] >= 90) &
            (df['ad_rating'].isin(['A+', 'A', 'A-', 'B+', 'B'])) &
            (df['composite_rating'] >= 80) &
            (df['avg_volume_50'] >= 100)
        ].copy()
        print(f"   Found {len(healthy_chart)} stocks")
        screens['healthy_chart'] = healthy_chart

        return screens

    def run_screener(self, output_dir: str = "screener_results") -> Dict[str, pd.DataFrame]:
        """Run the complete screener"""
        start_time = time.time()

        print("\n" + "="*60)
        print("IBD STYLE STOCK SCREENER")
        print("="*60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Get tickers
        tickers = self.get_tickers()

        # Fetch and analyze stocks
        df = self.fetch_and_analyze_stocks(tickers)

        if df.empty:
            print("\n❌ No stock data available")
            return {}

        # Apply screening criteria
        screens = self.apply_screening_criteria(df)

        # Save results
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*60)
        print("Saving results...")
        print("="*60)

        # Save complete dataset
        df.to_csv(f"{output_dir}/all_stocks_analyzed.csv", index=False)
        print(f"✓ Saved complete analysis: {output_dir}/all_stocks_analyzed.csv")

        # Save individual screen results
        for screen_name, screen_df in screens.items():
            if not screen_df.empty:
                filename = f"{output_dir}/{screen_name}.csv"
                screen_df.to_csv(filename, index=False)
                print(f"✓ Saved {screen_name}: {filename} ({len(screen_df)} stocks)")

        # Create summary report
        summary = []
        for screen_name, screen_df in screens.items():
            summary.append({
                'Screen': screen_name,
                'Count': len(screen_df),
                'Top_Stocks': ', '.join(screen_df.nlargest(5, 'rs_rating')['symbol'].tolist()) if len(screen_df) > 0 else ''
            })

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
        print(f"✓ Saved summary: {output_dir}/summary.csv")

        elapsed_time = time.time() - start_time

        print("\n" + "="*60)
        print("SCREENING COMPLETE")
        print("="*60)
        print(f"Total stocks analyzed: {len(df)}")
        print(f"Elapsed time: {elapsed_time/60:.2f} minutes")
        print(f"Results saved to: {output_dir}/")
        print("="*60)

        return screens


def main():
    """Main entry point"""
    try:
        screener = IBDScreener()
        screens = screener.run_screener()

        # Print summary
        print("\n" + "="*60)
        print("SCREEN SUMMARY")
        print("="*60)
        for screen_name, screen_df in screens.items():
            print(f"\n{screen_name.upper()}: {len(screen_df)} stocks")
            if len(screen_df) > 0:
                print("Top 5 by RS Rating:")
                top_5 = screen_df.nlargest(5, 'rs_rating')[['symbol', 'company_name', 'rs_rating', 'price', 'price_change_pct']]
                print(top_5.to_string(index=False))

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
