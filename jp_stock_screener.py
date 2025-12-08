"""
日本株専用スクリーナー (Japanese Stock Screener)

yfinanceを使用して日本株のスクリーニングを行います。
オリジナルのIBDスクリーナーのロジックを日本株向けに適合させています。

使用方法:
    python jp_stock_screener.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from curl_cffi import requests as cureq
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

class JPScreener:
    def __init__(self, tickers: List[str], benchmark_ticker: str = '^N225'):
        """
        Args:
            tickers: スクリーニング対象のティッカーリスト（例: ['7203.T', '9984.T']）
            benchmark_ticker: ベンチマーク（デフォルト: 日経平均 ^N225）
        """
        self.tickers = tickers
        self.benchmark_ticker = benchmark_ticker
        self.session = cureq.Session(impersonate="chrome110")
        self.data_cache = {}
        self.benchmark_data = None

        # ベンチマークデータの取得
        print(f"ベンチマークデータを取得中: {self.benchmark_ticker}...")
        try:
            bench = yf.Ticker(self.benchmark_ticker, session=self.session)
            self.benchmark_data = bench.history(period="1y")
        except Exception as e:
            print(f"ベンチマーク取得エラー: {e}")

    def fetch_data(self, ticker: str):
        """データを取得してキャッシュする"""
        if ticker in self.data_cache:
            return self.data_cache[ticker]

        try:
            t = yf.Ticker(ticker, session=self.session)

            # 履歴データ (1年分)
            history = t.history(period="1y")

            if history.empty:
                return None

            # 企業情報
            try:
                info = t.info
            except:
                info = {}

            # 財務データ (四半期)
            try:
                financials = t.quarterly_financials
            except:
                financials = None

            self.data_cache[ticker] = {
                'history': history,
                'info': info,
                'financials': financials
            }
            return self.data_cache[ticker]
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None

    # ==================== ヘルパーメソッド ====================

    def get_price_metrics(self, history: pd.DataFrame) -> Optional[Dict]:
        if history is None or len(history) < 2:
            return None

        close = history['Close'].values
        open_price = history['Open'].values

        metrics = {
            'price': close[-1],
            'pct_change_1d': ((close[-1] - close[-2]) / close[-2] * 100),
            'change_from_open': ((close[-1] - open_price[-1]) / open_price[-1] * 100) if open_price[-1] != 0 else 0,
            'pct_1m': None,
            'pct_3m': None,
            'pct_6m': None
        }

        if len(close) >= 21:
            metrics['pct_1m'] = ((close[-1] - close[-21]) / close[-21] * 100)
        if len(close) >= 63:
            metrics['pct_3m'] = ((close[-1] - close[-63]) / close[-63] * 100)
        if len(close) >= 126:
            metrics['pct_6m'] = ((close[-1] - close[-126]) / close[-126] * 100)

        return metrics

    def get_moving_averages(self, history: pd.DataFrame) -> Optional[Dict]:
        if history is None or len(history) < 200:
            return None
        close = history['Close'].values
        return {
            '10ma': np.mean(close[-10:]),
            '21ma': np.mean(close[-21:]),
            '50ma': np.mean(close[-50:]),
            '150ma': np.mean(close[-150:]),
            '200ma': np.mean(close[-200:]),
            'price': close[-1]
        }

    def get_volume_metrics(self, history: pd.DataFrame) -> Optional[Dict]:
        if history is None or len(history) < 90:
            return None
        volume = history['Volume'].values

        avg_vol_50 = np.mean(volume[-50:])
        avg_vol_90 = np.mean(volume[-90:])
        current = volume[-1]

        return {
            'current_volume': current,
            'avg_vol_50': avg_vol_50,
            'avg_vol_90': avg_vol_90,
            'rel_volume': current / avg_vol_50 if avg_vol_50 > 0 else 0,
            'vol_change_pct': ((current - avg_vol_50) / avg_vol_50 * 100) if avg_vol_50 > 0 else 0
        }

    def calculate_rs_rating(self, ticker_history: pd.DataFrame) -> float:
        """
        簡易的なRS Rating計算 (ベンチマークに対する相対強度)
        IBDの正確な計算式は非公開のため、1年間の相対パフォーマンスを使用
        """
        if self.benchmark_data is None or ticker_history is None:
            return 0

        # 日付を合わせて結合
        df = pd.merge(
            self.benchmark_data[['Close']].rename(columns={'Close': 'bench'}),
            ticker_history[['Close']].rename(columns={'Close': 'stock'}),
            left_index=True, right_index=True, how='inner'
        )

        if len(df) < 252:
            return 0 # データ不足

        # 1年間のパフォーマンス比較
        stock_perf = (df['stock'].iloc[-1] / df['stock'].iloc[0]) - 1
        bench_perf = (df['bench'].iloc[-1] / df['bench'].iloc[0]) - 1

        # ベンチマークよりどれだけ良いか
        return (stock_perf - bench_perf) * 100

    # ==================== スクリーナー実装 ====================

    def screener_momentum_97(self) -> List[str]:
        """Momentum 97: 直近のパフォーマンスが非常に高い銘柄"""
        print("  実行中: Momentum 97...")
        candidates = []
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data: continue

            m = self.get_price_metrics(data['history'])
            if m and m['pct_1m'] and m['pct_3m'] and m['pct_6m']:
                # 簡易判定: すべての期間でプラス、かつ直近3ヶ月で20%以上上昇 (閾値は調整可能)
                if m['pct_1m'] > 0 and m['pct_3m'] > 20 and m['pct_6m'] > 30:
                    candidates.append(ticker)
        return candidates

    def screener_explosive_eps_growth(self) -> List[str]:
        """Explosive EPS Growth: 四半期EPSが急成長"""
        print("  実行中: Explosive EPS Growth...")
        candidates = []
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data: continue

            # EPSチェック
            fin = data['financials']
            if fin is not None and not fin.empty and 'Basic EPS' in fin.index:
                eps = fin.loc['Basic EPS'].values
                # 最新と前回の比較 (降順前提)
                if len(eps) >= 2 and eps[1] != 0:
                    growth = ((eps[0] - eps[1]) / abs(eps[1])) * 100
                    if growth > 50: # 50%以上成長

                        # テクニカルチェックも追加
                        ma = self.get_moving_averages(data['history'])
                        if ma and ma['price'] >= ma['50ma']: # 50日線以上
                            candidates.append(ticker)
        return candidates

    def screener_up_on_volume(self) -> List[str]:
        """Up on Volume: 出来高を伴って上昇"""
        print("  実行中: Up on Volume...")
        candidates = []
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data: continue

            pm = self.get_price_metrics(data['history'])
            vm = self.get_volume_metrics(data['history'])

            if pm and vm:
                # 価格上昇、出来高増加
                if pm['pct_change_1d'] > 0 and vm['vol_change_pct'] > 20 and vm['avg_vol_50'] > 100000:
                    candidates.append(ticker)
        return candidates

    def screener_top_rs_rating(self) -> List[str]:
        """Top RS Rating: 相対強度が強い"""
        print("  実行中: Top RS Rating (Simulated)...")
        candidates = []
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data: continue

            rs_score = self.calculate_rs_rating(data['history'])

            # ベンチマークを20%以上アウトパフォーム
            if rs_score > 20:
                ma = self.get_moving_averages(data['history'])
                if ma and ma['10ma'] > ma['21ma'] > ma['50ma']: # 上昇トレンド
                    candidates.append(ticker)
        return candidates

    def screener_bullish_yesterday(self) -> List[str]:
        """4% Bullish Yesterday: 昨日4%以上上昇"""
        print("  実行中: 4% Bullish Yesterday...")
        candidates = []
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data: continue

            pm = self.get_price_metrics(data['history'])
            vm = self.get_volume_metrics(data['history'])

            if pm and vm:
                if (pm['pct_change_1d'] > 4 and
                    pm['change_from_open'] > 0 and
                    vm['rel_volume'] > 1.2):
                    candidates.append(ticker)
        return candidates

    def screener_healthy_chart(self) -> List[str]:
        """Healthy Chart: 健全なチャート形状 (パーフェクトオーダー)"""
        print("  実行中: Healthy Chart...")
        candidates = []
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data: continue

            ma = self.get_moving_averages(data['history'])
            if ma:
                # パーフェクトオーダー (10 > 21 > 50 > 150 > 200)
                if (ma['10ma'] > ma['21ma'] > ma['50ma'] > ma['150ma'] > ma['200ma']):
                    candidates.append(ticker)
        return candidates

    def run_all(self):
        print(f"\n=== 日本株スクリーナー実行 (対象: {len(self.tickers)}銘柄) ===")

        results = {}
        results['Momentum 97'] = self.screener_momentum_97()
        results['Explosive EPS'] = self.screener_explosive_eps_growth()
        results['Up on Volume'] = self.screener_up_on_volume()
        results['Top RS Rating'] = self.screener_top_rs_rating()
        results['Bullish Yesterday'] = self.screener_bullish_yesterday()
        results['Healthy Chart'] = self.screener_healthy_chart()

        print("\n" + "="*50)
        print("スクリーニング結果")
        print("="*50)

        for name, tickers in results.items():
            print(f"\n【{name}】: {len(tickers)}件")
            if tickers:
                print(f"  {', '.join(tickers)}")
            else:
                print("  該当なし")
        print("\n" + "="*50)

def main():
    # テスト用の銘柄リスト (実際には東証全銘柄などを読み込む必要があります)
    # ここでは代表的な銘柄と、動きのありそうな銘柄をミックス
    tickers = [
        "7203.T", "9984.T", "6758.T", "8035.T", "6861.T", # 大型
        "6920.T", "8306.T", "6146.T", "7011.T", "7012.T", # 半導体・防衛・銀行
        "9101.T", "9104.T", "9107.T", # 海運
        "5401.T", "8058.T"  # 鉄鋼・商社
    ]

    screener = JPScreener(tickers)
    screener.run_all()

if __name__ == "__main__":
    main()
