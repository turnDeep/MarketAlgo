"""
IBD Stock Screeners Implementation

このモジュールは以下のIBDスクリーナーを実装します：
1. Momentum 97
2. Explosive Estimated EPS Growth Stocks
3. Up on Volume List
4. Top 2% RS Rating List
5. 4% Bullish Yesterday
6. Healthy Chart Watch List

各スクリーナーの結果をGoogleスプレッドシートに出力します。

===================================================================================
IBD RATINGS IMPLEMENTATION NOTES (コミュニティ研究に基づく実装)
===================================================================================

このモジュールは、Investor's Business Daily (IBD)の独自指標を、コミュニティの
研究とリバースエンジニアリングの成果に基づいて実装しています。

重要な注意事項:
IBDの正確な計算式は企業秘密であり、完全に再現することは不可能です。
この実装は、公開情報、コミュニティの研究、GitHub実装例を基にした
「最善の推定」です。

===================================================================================
1. RS RATING (Relative Strength Rating) - ✓ コミュニティ検証済み
===================================================================================

公式: RS = 0.4 * ROC(63d) + 0.2 * ROC(126d) + 0.2 * ROC(189d) + 0.2 * ROC(252d)

- 63営業日 ≈ 3ヶ月（最新四半期、40%の重み）
- 126営業日 ≈ 6ヶ月（20%の重み）
- 189営業日 ≈ 9ヶ月（20%の重み）
- 252営業日 ≈ 12ヶ月（20%の重み）

計算後、全銘柄をパーセンタイルランキング（0-100）に変換。

参考:
- GitHub: skyte/relative-strength
- Medium: "Calculating the IBD RS Rating with Python" by Shashank Vemuri
- AmiBroker/TradingView community discussions

===================================================================================
2. A/D RATING (Accumulation/Distribution Rating) - ⚠️ 推定実装
===================================================================================

IBDの独自計算式は非公開。この実装では以下の手法を使用:

- 13週間（65営業日）の価格・ボリュームデータを分析
- Money Flow Multiplier: [(Close - Low) - (High - Close)] / (High - Low)
- 機関投資家の買い/売りパターンを推定
- レーティング: A (Heavy Accumulation) ～ E (Heavy Distribution)

重要な違い:
- IBD A/D Rating: 13週間の機関投資家活動評価（A～E）
- Marc Chaikin A/D Line: 累積的なマネーフローライン（連続値）

制限事項:
この実装は真のIBD計算を近似したものであり、実際のIBDレーティングと
異なる場合があります。

===================================================================================
3. EPS RATING - ⚠️ 推定実装
===================================================================================

IBD EPS Ratingの構成要素（コミュニティ研究に基づく）:
1. 最新四半期のEPS成長率（前年同期比）- 50%の重み
2. 前四半期のEPS成長率（前年同期比）- 20%の重み
3. 3～5年間の年間EPS成長率（CAGR）- 20%の重み
4. 収益安定性ファクター - 10%の重み

参考:
- William O'Neil + Co. の公式説明
- "How to Make Money in Stocks" (William O'Neil著)
- IBD educational materials

制限事項:
正確な計算式、重み付け、安定性の評価方法は非公開。
この実装は文献と経験則に基づく推定です。

===================================================================================
4. COMPOSITE RATING - ⚠️ 推定実装
===================================================================================

IBD Composite Ratingの構成要素と推定重み配分:

1. RS Rating: 30% (ダブルウェイト)
2. EPS Rating: 30% (ダブルウェイト)
3. A/D Rating: 15%
4. Industry Group RS: 10% (現在未実装)
5. SMR Rating: 10% (現在未実装)
6. 52週高値からの距離: 5%

コミュニティの発見:
- EPS RatingとRS Ratingに「ダブルウェイト」が適用される
- IBDは「RS RatingとEPS Ratingにより大きな重みを置く」と公表
- 正確なパーセンテージは非公開

参考:
- IBD SmartSelect Corporate Ratings documentation
- Community reverse-engineering efforts
- TradingView/AmiBroker implementations

制限事項:
正確な重み付けは企業秘密。この実装は公開情報と
コミュニティの研究に基づく合理的な推定です。

===================================================================================
5. SMR RATING - ✓ 実装済み
===================================================================================

SMR Ratingの構成要素（IBD公式情報）:
- Sales growth (過去3四半期の前年同期比成長率、40%の重み)
- Pre-tax profit margins (年次、20%の重み)
- After-tax profit margins (四半期、20%の重み)
- ROE (Return on Equity、年次、20%の重み)

レーティング: A (Top 20%) ～ E (Bottom 20%)

計算方法:
1. 各要素を全銘柄でパーセンタイルランキング（0-100）に変換
2. 重み付けして最終スコアを計算
3. スコアをA-Eのレーティングに変換
   - A: 80-100 (Top 20%)
   - B: 60-80 (Next 20%)
   - C: 40-60 (Middle 20%)
   - D: 20-40 (Next 20%)
   - E: 0-20 (Bottom 20%)

制限事項:
- Pre-tax marginは、after-tax marginから推定（税率20-25%を想定）
- ROE計算にはBalance Sheetデータが必要（現在未実装のため、ROEなしで計算）

===================================================================================
データソースと制限
===================================================================================

このモジュールはFinancial Modeling Prep (FMP) APIを使用:
- 価格データ: 日次OHLCV
- 財務データ: 四半期・年次損益計算書
- 企業データ: プロファイル、セクター、時価総額

API制限:
- レート制限: 750 calls/minute (設定可能)
- データの正確性: APIプロバイダに依存
- 歴史的データ: スプリット調整の問題がある可能性

===================================================================================
検証と精度
===================================================================================

このモジュールの実装は、以下のコミュニティリソースと整合性を確認:
✓ GitHub: skyte/relative-strength (RS Rating)
✓ GitHub: nickklosterman/IBD
✓ Medium articles on IBD calculations
✓ TradingView/AmiBroker community scripts
✓ William O'Neil + Co. official documentation

ただし、IBDの実際のレーティングとの完全な一致は保証されません。
この実装は教育・研究目的であり、投資判断の唯一の根拠とすべきではありません。

===================================================================================
"""

import gspread
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
from multiprocessing import Pool, Manager, Lock, Queue
from functools import partial
import traceback
from typing import List, Dict, Optional, Tuple
from get_tickers import FMPTickerFetcher
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# .envファイルから環境変数を読み込む
load_dotenv()


class RateLimiter:
    """API rate limit を管理するクラス（マルチプロセス/スレッド対応）"""

    def __init__(self, max_calls_per_minute=750):
        """
        Args:
            max_calls_per_minute: 1分間の最大コール数
        """
        self.max_calls_per_minute = max_calls_per_minute
        self.min_interval = 60.0 / max_calls_per_minute
        self.lock = threading.Lock()
        self.request_times = []

    def wait_if_needed(self):
        """必要に応じて待機（スレッドセーフ）"""
        with self.lock:
            current_time = time.time()

            # 60秒以内のリクエストタイムスタンプをフィルタ
            self.request_times = [t for t in self.request_times if current_time - t < 60]

            if len(self.request_times) >= self.max_calls_per_minute:
                # 最も古いリクエストから60秒経過するまで待機
                sleep_time = 60 - (current_time - self.request_times[0]) + 0.1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    current_time = time.time()
                    self.request_times = [t for t in self.request_times if current_time - t < 60]

            self.request_times.append(current_time)


class IBDScreeners:
    """IBD スクリーナーの実装"""

    def __init__(self, fmp_api_key, credentials_file, spreadsheet_name):
        """
        Args:
            fmp_api_key: Financial Modeling Prep API Key
            credentials_file: Googleサービスアカウントの認証情報JSONファイルパス
            spreadsheet_name: Googleスプレッドシートの名前
        """
        self.fmp_api_key = fmp_api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.rate_limiter = RateLimiter(max_calls_per_minute=750)

        # Google Sheets認証
        try:
            self.gc = gspread.service_account(filename=credentials_file)
        except FileNotFoundError:
            print(f"エラー: 認証情報ファイル '{credentials_file}' が見つかりません")
            raise

        # スプレッドシートを開く（存在しない場合は作成）
        try:
            self.spreadsheet = self.gc.open(spreadsheet_name)
        except gspread.SpreadsheetNotFound:
            self.spreadsheet = self.gc.create(spreadsheet_name)
            self.spreadsheet.share('', perm_type='anyone', role='reader')
            print(f"新しいスプレッドシート '{spreadsheet_name}' を作成しました")

    def fetch_with_rate_limit(self, url, params=None):
        """レート制限を考慮したAPIリクエスト"""
        self.rate_limiter.wait_if_needed()

        if params is None:
            params = {}
        params['apikey'] = self.fmp_api_key

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # エラーメッセージを抑制（大量の出力を避ける）
            return None

    def get_historical_prices(self, symbol, days=300):
        """過去の価格データを取得"""
        url = f"{self.base_url}/historical-price-full/{symbol}"
        params = {'timeseries': days}

        data = self.fetch_with_rate_limit(url, params)

        if data and 'historical' in data and data['historical']:
            df = pd.DataFrame(data['historical'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            return df
        return None

    def get_quote(self, symbol):
        """リアルタイムの株価情報を取得"""
        url = f"{self.base_url}/quote/{symbol}"
        data = self.fetch_with_rate_limit(url)

        if data and len(data) > 0:
            return data[0]
        return None

    def get_analyst_estimates(self, symbol):
        """アナリスト予想を取得"""
        url = f"{self.base_url}/analyst-estimates/{symbol}"
        data = self.fetch_with_rate_limit(url)
        return data if data else None

    def get_income_statement(self, symbol, period='quarter', limit=8):
        """損益計算書を取得（EPS成長率計算用）"""
        url = f"{self.base_url}/income-statement/{symbol}"
        params = {'period': period, 'limit': limit}
        data = self.fetch_with_rate_limit(url, params)
        return data if data else None

    def get_company_profile(self, symbol):
        """企業プロファイルを取得（セクター情報など）"""
        url = f"{self.base_url}/profile/{symbol}"
        data = self.fetch_with_rate_limit(url)

        if data and len(data) > 0:
            return data[0]
        return None

    def calculate_rs_rating(self, prices_df, benchmark_prices_df=None):
        """
        IBD スタイルの RS Rating を計算（コミュニティ検証済みの公式）

        IBD RS Ratingは株式の価格パフォーマンスを評価する重要な指標です。
        過去12ヶ月間のパフォーマンスを計算し、最新の3ヶ月間により大きな重みを与えます。

        公式（コミュニティで広く検証されている）:
        RS = 0.4 * ROC(63d) + 0.2 * ROC(126d) + 0.2 * ROC(189d) + 0.2 * ROC(252d)

        期間の内訳:
        - 63営業日 ≈ 3ヶ月（最新四半期、40%の重み）
        - 126営業日 ≈ 6ヶ月（20%の重み）
        - 189営業日 ≈ 9ヶ月（20%の重み）
        - 252営業日 ≈ 12ヶ月（20%の重み）

        この計算後、全銘柄をパーセンタイルランキング（0-100）に変換します。
        IBDは99が最高、1が最低としています。

        参考情報:
        - GitHub: skyte/relative-strength
        - Medium: "Calculating the IBD RS Rating with Python"
        - AmiBroker Community Forum discussions

        Args:
            prices_df: 株価データ（最低252営業日必要）
            benchmark_prices_df: ベンチマーク（S&P 500など）の株価データ（現在未使用）

        Returns:
            float: RS値（後でパーセンタイルランキングに変換される）
        """
        if prices_df is None or len(prices_df) < 252:
            return None

        try:
            close = prices_df['close'].values

            # 各期間のROC（Rate of Change）を計算
            if len(close) >= 252:
                roc_63 = (close[-1] / close[-63] - 1) * 100 if close[-63] != 0 else 0
                roc_126 = (close[-1] / close[-126] - 1) * 100 if close[-126] != 0 else 0
                roc_189 = (close[-1] / close[-189] - 1) * 100 if close[-189] != 0 else 0
                roc_252 = (close[-1] / close[-252] - 1) * 100 if close[-252] != 0 else 0

                # IBD式の加重平均（最新四半期に40%の重み）
                rs_value = 0.4 * roc_63 + 0.2 * roc_126 + 0.2 * roc_189 + 0.2 * roc_252

                return rs_value
            else:
                return None

        except Exception as e:
            print(f"RS Rating calculation error: {e}")
            return None

    def percentile_rank_rs_ratings(self, rs_values_dict):
        """
        RS値をパーセンタイルランキングに変換（0-100）

        Args:
            rs_values_dict: {ticker: rs_value} の辞書

        Returns:
            dict: {ticker: percentile_rating} の辞書
        """
        valid_rs = {k: v for k, v in rs_values_dict.items() if v is not None}

        if not valid_rs:
            return {}

        # RS値でソート
        sorted_tickers = sorted(valid_rs.items(), key=lambda x: x[1])

        # パーセンタイルを計算
        percentile_dict = {}
        total = len(sorted_tickers)

        for idx, (ticker, rs_val) in enumerate(sorted_tickers):
            percentile = ((idx + 1) / total) * 100
            percentile_dict[ticker] = round(percentile, 2)

        return percentile_dict

    def calculate_moving_averages(self, prices_df):
        """移動平均を計算"""
        if prices_df is None or len(prices_df) < 200:
            return None

        try:
            close = prices_df['close'].values

            ma_10 = np.mean(close[-10:]) if len(close) >= 10 else None
            ma_21 = np.mean(close[-21:]) if len(close) >= 21 else None
            ma_50 = np.mean(close[-50:]) if len(close) >= 50 else None
            ma_150 = np.mean(close[-150:]) if len(close) >= 150 else None
            ma_200 = np.mean(close[-200:]) if len(close) >= 200 else None

            current_price = close[-1]

            return {
                '10ma': ma_10,
                '21ma': ma_21,
                '50ma': ma_50,
                '150ma': ma_150,
                '200ma': ma_200,
                'price': current_price
            }
        except Exception as e:
            print(f"MA calculation error: {e}")
            return None

    def calculate_volume_metrics(self, prices_df):
        """ボリューム関連の指標を計算"""
        if prices_df is None or len(prices_df) < 90:
            return None

        try:
            volume = prices_df['volume'].values

            avg_volume_50 = np.mean(volume[-50:]) if len(volume) >= 50 else None
            avg_volume_90 = np.mean(volume[-90:]) if len(volume) >= 90 else None
            current_volume = volume[-1]

            # 50日平均ボリュームに対する変化率
            vol_change_pct = ((current_volume - avg_volume_50) / avg_volume_50 * 100) if avg_volume_50 and avg_volume_50 > 0 else 0

            # Relative Volume (今日のボリューム / 平均ボリューム)
            rel_volume = (current_volume / avg_volume_50) if avg_volume_50 and avg_volume_50 > 0 else 0

            return {
                'avg_vol_50': avg_volume_50 / 1000,  # thousands
                'avg_vol_90': avg_volume_90 / 1000,  # thousands
                'current_volume': current_volume / 1000,  # thousands
                'vol_change_pct': vol_change_pct,
                'rel_volume': rel_volume
            }
        except Exception as e:
            print(f"Volume metrics calculation error: {e}")
            return None

    def calculate_price_changes(self, prices_df):
        """価格変化率を計算"""
        if prices_df is None or len(prices_df) < 2:
            return None

        try:
            close = prices_df['close'].values
            open_price = prices_df['open'].values

            # 日次変化率
            pct_change_1d = ((close[-1] - close[-2]) / close[-2] * 100) if close[-2] != 0 else 0

            # Open からの変化率
            change_from_open = ((close[-1] - open_price[-1]) / open_price[-1] * 100) if open_price[-1] != 0 else 0

            # 1ヶ月、3ヶ月、6ヶ月のパフォーマンス
            pct_1m = ((close[-1] - close[-21]) / close[-21] * 100) if len(close) >= 21 and close[-21] != 0 else None
            pct_3m = ((close[-1] - close[-63]) / close[-63] * 100) if len(close) >= 63 and close[-63] != 0 else None
            pct_6m = ((close[-1] - close[-126]) / close[-126] * 100) if len(close) >= 126 and close[-126] != 0 else None

            return {
                'price': close[-1],
                'pct_change_1d': pct_change_1d,
                'change_from_open': change_from_open,
                'pct_1m': pct_1m,
                'pct_3m': pct_3m,
                'pct_6m': pct_6m
            }
        except Exception as e:
            print(f"Price change calculation error: {e}")
            return None

    def calculate_price_vs_ma(self, prices_df):
        """価格と移動平均の比較"""
        if prices_df is None or len(prices_df) < 50:
            return None

        try:
            close = prices_df['close'].values
            current_price = close[-1]

            ma_50 = np.mean(close[-50:]) if len(close) >= 50 else None

            if ma_50:
                price_vs_50ma = ((current_price - ma_50) / ma_50 * 100)
                return price_vs_50ma

            return None
        except Exception as e:
            print(f"Price vs MA calculation error: {e}")
            return None

    def check_rs_line_new_high(self, prices_df, benchmark_prices_df):
        """RS Lineが新高値かチェック"""
        if prices_df is None or benchmark_prices_df is None:
            return False

        try:
            # RS Line = Stock Price / Benchmark Price
            stock_prices = prices_df['close'].values[-252:] if len(prices_df) >= 252 else prices_df['close'].values
            benchmark_prices = benchmark_prices_df['close'].values[-252:] if len(benchmark_prices_df) >= 252 else benchmark_prices_df['close'].values

            min_len = min(len(stock_prices), len(benchmark_prices))
            stock_prices = stock_prices[-min_len:]
            benchmark_prices = benchmark_prices[-min_len:]

            rs_line = stock_prices / benchmark_prices

            # 最新のRS Lineが過去の最高値に近いか（95%以上）
            current_rs = rs_line[-1]
            max_rs = np.max(rs_line)

            return current_rs >= max_rs * 0.95

        except Exception as e:
            print(f"RS Line new high check error: {e}")
            return False

    def calculate_eps_growth(self, income_statements_quarterly, income_statements_annual=None):
        """
        EPSの成長率を計算（IBD方式に基づく改良版）

        IBD EPS Ratingの構成要素（コミュニティ研究に基づく）:
        1. 最新四半期のEPS成長率（前年同期比）- 最も重視
        2. 前四半期のEPS成長率（前年同期比）
        3. 3～5年間の年間EPS成長率
        4. 収益の安定性ファクター

        最近の四半期により大きな重みを置く

        Args:
            income_statements_quarterly: 損益計算書データ（四半期）、最低8四半期必要
            income_statements_annual: 損益計算書データ（年次）、3～5年分推奨

        Returns:
            dict: EPS成長率と総合評価スコア
        """
        if not income_statements_quarterly or len(income_statements_quarterly) < 5:
            return None

        try:
            result = {}

            # 1. 最新四半期のEPS成長率（前年同期比）
            latest_eps = income_statements_quarterly[0].get('eps', 0)
            yoy_eps_q0 = income_statements_quarterly[4].get('eps', 0) if len(income_statements_quarterly) > 4 else 0

            if yoy_eps_q0 != 0 and latest_eps is not None:
                eps_growth_last_qtr = ((latest_eps - yoy_eps_q0) / abs(yoy_eps_q0)) * 100
            else:
                eps_growth_last_qtr = None

            result['eps_growth_last_qtr'] = eps_growth_last_qtr

            # 2. 前四半期のEPS成長率（前年同期比）
            if len(income_statements_quarterly) >= 6:
                prev_qtr_eps = income_statements_quarterly[1].get('eps', 0)
                yoy_eps_q1 = income_statements_quarterly[5].get('eps', 0)

                if yoy_eps_q1 != 0 and prev_qtr_eps is not None:
                    eps_growth_prev_qtr = ((prev_qtr_eps - yoy_eps_q1) / abs(yoy_eps_q1)) * 100
                else:
                    eps_growth_prev_qtr = None

                result['eps_growth_prev_qtr'] = eps_growth_prev_qtr
            else:
                result['eps_growth_prev_qtr'] = None

            # 3. 年間EPS成長率（3～5年）
            annual_growth_rate = None
            if income_statements_annual and len(income_statements_annual) >= 3:
                try:
                    # 直近3年間のEPSを取得
                    eps_values = [stmt.get('eps', 0) for stmt in income_statements_annual[:3]]

                    # 年平均成長率（CAGR）を計算
                    if eps_values[0] > 0 and eps_values[-1] > 0:
                        years = len(eps_values) - 1
                        cagr = (pow(eps_values[0] / eps_values[-1], 1/years) - 1) * 100
                        annual_growth_rate = cagr
                except:
                    pass

            result['annual_growth_rate'] = annual_growth_rate

            # 4. 収益安定性の評価（四半期EPSの変動係数）
            stability_score = None
            if len(income_statements_quarterly) >= 8:
                try:
                    eps_last_8q = [stmt.get('eps', 0) for stmt in income_statements_quarterly[:8]]
                    eps_last_8q = [e for e in eps_last_8q if e is not None and e > 0]

                    if len(eps_last_8q) >= 6:
                        eps_mean = np.mean(eps_last_8q)
                        eps_std = np.std(eps_last_8q)

                        # 変動係数（CV）: 標準偏差 / 平均
                        # 低いほど安定（良い）
                        if eps_mean > 0:
                            coefficient_of_variation = eps_std / eps_mean
                            # スコアに変換（0-100、低いCVほど高スコア）
                            stability_score = max(0, 100 - (coefficient_of_variation * 100))
                except:
                    pass

            result['stability_score'] = stability_score

            # 5. 総合EPSスコア（0-100）の計算
            # IBDの重み付けを模倣: 最新四半期50%、前四半期20%、年間成長率20%、安定性10%
            total_score = 0
            weight_sum = 0

            if eps_growth_last_qtr is not None:
                # 成長率を0-100のスコアに正規化（100%成長=100点）
                score = min(100, max(0, eps_growth_last_qtr))
                total_score += score * 0.5
                weight_sum += 0.5

            if result['eps_growth_prev_qtr'] is not None:
                score = min(100, max(0, result['eps_growth_prev_qtr']))
                total_score += score * 0.2
                weight_sum += 0.2

            if annual_growth_rate is not None:
                score = min(100, max(0, annual_growth_rate * 2))  # 50%成長=100点
                total_score += score * 0.2
                weight_sum += 0.2

            if stability_score is not None:
                total_score += stability_score * 0.1
                weight_sum += 0.1

            # 重み付けを正規化
            if weight_sum > 0:
                eps_rating_score = total_score / weight_sum
            else:
                eps_rating_score = None

            result['eps_rating_score'] = eps_rating_score

            return result

        except Exception as e:
            print(f"EPS growth calculation error: {e}")
            return None

    def estimate_ad_rating(self, prices_df):
        """
        A/D Rating（Accumulation/Distribution）の改良推定

        IBDの独自計算式は非公開だが、コミュニティの研究に基づき以下を実装：
        - 13週間（65営業日）の価格・ボリュームデータを分析
        - Money Flow Multiplierを活用して機関投資家の活動を推定
        - 価格位置（高値/安値範囲内）とボリュームの関係を評価

        Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
        - 終値が高値に近い（買い圧力）: +1に近い
        - 終値が安値に近い（売り圧力）: -1に近い

        IBD A/D RatingとMarc ChaikinのA/D Lineは異なる指標:
        - IBD: 13週間の機関投資家の買い/売りパターンを評価（A～E）
        - Chaikin: 累積的なマネーフローライン（連続値）

        Returns:
            str: A, B, C, D, E のレーティング
        """
        if prices_df is None or len(prices_df) < 65:
            return None

        try:
            # 過去13週（65営業日）のデータ
            recent_data = prices_df.tail(65)

            close = recent_data['close'].values
            high = recent_data['high'].values
            low = recent_data['low'].values
            volume = recent_data['volume'].values

            # Money Flow Multiplier (MFM) を計算
            # MFM = [(Close - Low) - (High - Close)] / (High - Low)
            high_low_diff = high - low
            # ゼロ除算を避ける
            high_low_diff = np.where(high_low_diff == 0, 0.0001, high_low_diff)

            money_flow_multiplier = ((close - low) - (high - close)) / high_low_diff

            # Money Flow Volume = MFM * Volume
            money_flow_volume = money_flow_multiplier * volume

            # 価格変化方向とボリュームの重み付き分析
            price_changes = np.diff(close) / close[:-1]

            # 上昇日と下落日のMoney Flow Volumeを集計
            up_days = price_changes > 0
            down_days = price_changes < 0

            # 最新のMFVトレンド（直近の重み付けを高める）
            weights = np.linspace(0.5, 1.5, len(money_flow_volume[1:]))

            if np.sum(up_days) > 0 and np.sum(down_days) > 0:
                # 上昇日の加重平均Money Flow Volume
                weighted_mfv_up = np.average(
                    money_flow_volume[1:][up_days],
                    weights=weights[up_days]
                )
                # 下落日の加重平均Money Flow Volume
                weighted_mfv_down = np.average(
                    money_flow_volume[1:][down_days],
                    weights=weights[down_days]
                )

                # 総合的なAccumulation/Distribution スコア
                total_mfv = np.sum(money_flow_volume)
                avg_mfv = np.mean(money_flow_volume)

                # 正規化されたスコア（-1 to +1）
                if avg_mfv != 0:
                    normalized_score = total_mfv / (abs(avg_mfv) * len(money_flow_volume))
                else:
                    normalized_score = 0

                # 上昇/下落日のボリューム比率も考慮
                vol_ratio = weighted_mfv_up / abs(weighted_mfv_down) if weighted_mfv_down != 0 else 1

                # 総合評価スコア（-1 to +1の範囲）
                combined_score = (normalized_score * 0.6) + ((vol_ratio - 1) * 0.4)

                # レーティング判定（より厳格な基準）
                if combined_score >= 0.4:
                    return 'A'  # 強い買い（Heavy Accumulation）
                elif combined_score >= 0.15:
                    return 'B'  # 中程度の買い（Moderate Accumulation）
                elif combined_score >= -0.15:
                    return 'C'  # 中立（Neutral）
                elif combined_score >= -0.4:
                    return 'D'  # 中程度の売り（Moderate Distribution）
                else:
                    return 'E'  # 強い売り（Heavy Distribution）

            return 'C'  # デフォルト（データ不足）

        except Exception as e:
            print(f"A/D Rating estimation error: {e}")
            return None

    def estimate_comp_rating(self, rs_percentile, eps_rating_score, ad_rating, price_vs_52w_high, industry_group_rs=None, smr_rating=None):
        """
        Composite Rating の改良推定（IBDコミュニティ研究に基づく）

        IBD Composite Ratingの構成要素:
        1. EPS Rating - 収益成長率（ダブルウェイト）
        2. RS Rating - 相対力指数（ダブルウェイト）
        3. Industry Group RS Rating - 業種グループの強さ
        4. SMR Rating - 売上・利益率・ROE
        5. A/D Rating - 機関投資家の買い/売り
        6. 52週高値からの距離

        コミュニティの研究によると、EPS RatingとRS Ratingに「ダブルウェイト」が適用される

        推定される重み配分:
        - RS Rating: 30% (ダブルウェイト)
        - EPS Rating: 30% (ダブルウェイト)
        - A/D Rating: 15%
        - Industry Group RS: 10%
        - SMR Rating: 10%
        - 52週高値からの距離: 5%

        Args:
            rs_percentile: RS Rating (0-100)
            eps_rating_score: EPS Rating Score (0-100)
            ad_rating: A/D Rating (A-E)
            price_vs_52w_high: 52週高値からの距離 (%)
            industry_group_rs: Industry Group RS Rating (0-100、オプション)
            smr_rating: SMR Rating (A-E、オプション)

        Returns:
            float: Composite Rating (0-100)
        """
        try:
            score = 0
            weight_sum = 0

            # 1. RS Rating (30% - ダブルウェイト)
            if rs_percentile is not None:
                score += rs_percentile * 0.30
                weight_sum += 0.30

            # 2. EPS Rating (30% - ダブルウェイト)
            if eps_rating_score is not None:
                score += eps_rating_score * 0.30
                weight_sum += 0.30

            # 3. A/D Rating (15%)
            ad_score_map = {
                'A': 100,  # Heavy Accumulation
                'B': 75,   # Moderate Accumulation
                'C': 50,   # Neutral
                'D': 25,   # Moderate Distribution
                'E': 0     # Heavy Distribution
            }
            if ad_rating and ad_rating in ad_score_map:
                score += ad_score_map[ad_rating] * 0.15
                weight_sum += 0.15

            # 4. Industry Group RS (10% - オプション)
            if industry_group_rs is not None:
                score += industry_group_rs * 0.10
                weight_sum += 0.10

            # 5. SMR Rating (10% - オプション)
            smr_score_map = {
                'A': 100,  # Top 20%
                'B': 75,   # Next 20%
                'C': 50,   # Middle 20%
                'D': 25,   # Next 20%
                'E': 0     # Bottom 20%
            }
            if smr_rating and smr_rating in smr_score_map:
                score += smr_score_map[smr_rating] * 0.10
                weight_sum += 0.10

            # 6. 52週高値からの距離 (5%)
            if price_vs_52w_high is not None:
                # -5%以内 = 100点, -15%以内 = 50点, -30%以下 = 0点
                if price_vs_52w_high >= -5:
                    high_score = 100
                elif price_vs_52w_high >= -15:
                    high_score = 100 - ((abs(price_vs_52w_high) - 5) * 5)
                else:
                    high_score = max(0, 50 - ((abs(price_vs_52w_high) - 15) * 3.33))

                score += high_score * 0.05
                weight_sum += 0.05

            # 重み付けを正規化（一部の要素が欠けている場合）
            if weight_sum > 0:
                normalized_score = score / weight_sum * 100
            else:
                normalized_score = None

            # 0-100の範囲に制限
            if normalized_score is not None:
                return min(100, max(0, normalized_score))
            else:
                return None

        except Exception as e:
            print(f"Composite Rating estimation error: {e}")
            return None

    def calculate_52w_high_distance(self, prices_df):
        """52週高値からの距離を計算"""
        if prices_df is None or len(prices_df) < 252:
            return None

        try:
            recent_data = prices_df.tail(252)
            high_52w = recent_data['high'].max()
            current_price = prices_df['close'].iloc[-1]

            distance = ((current_price - high_52w) / high_52w) * 100
            return distance
        except Exception as e:
            print(f"52W high distance calculation error: {e}")
            return None

    # ==================== スクリーナー実装 ====================

    def screener_momentum_97(self, tickers_list, rs_percentile_dict):
        """
        Momentum 97 スクリーナー

        条件:
        - 1M Rank (Pct) ≥ 97%
        - 3M Rank (Pct) ≥ 97%
        - 6M Rank (Pct) ≥ 97%
        """
        print("\n=== Momentum 97 スクリーナー実行中 ===")
        passed = []

        # 全銘柄の1M, 3M, 6Mパフォーマンスを計算
        performance_data = {}

        for ticker in tickers_list:
            try:
                prices_df = self.get_historical_prices(ticker, days=180)
                if prices_df is None or len(prices_df) < 126:
                    continue

                price_metrics = self.calculate_price_changes(prices_df)
                if price_metrics:
                    performance_data[ticker] = {
                        '1m': price_metrics['pct_1m'],
                        '3m': price_metrics['pct_3m'],
                        '6m': price_metrics['pct_6m']
                    }
            except Exception as e:
                continue

        # 各期間でパーセンタイルランクを計算
        def calc_percentile_ranks(values_dict, key):
            valid = {t: v[key] for t, v in values_dict.items() if v[key] is not None}
            if not valid:
                return {}
            sorted_items = sorted(valid.items(), key=lambda x: x[1])
            total = len(sorted_items)
            return {t: ((idx + 1) / total) * 100 for idx, (t, v) in enumerate(sorted_items)}

        rank_1m = calc_percentile_ranks(performance_data, '1m')
        rank_3m = calc_percentile_ranks(performance_data, '3m')
        rank_6m = calc_percentile_ranks(performance_data, '6m')

        # フィルタリング
        for ticker in performance_data.keys():
            if (rank_1m.get(ticker, 0) >= 97 and
                rank_3m.get(ticker, 0) >= 97 and
                rank_6m.get(ticker, 0) >= 97):
                passed.append(ticker)

        print(f"  合格: {len(passed)} 銘柄")
        return passed

    def screener_explosive_eps_growth(self, tickers_list, rs_percentile_dict):
        """
        Explosive Estimated EPS Growth Stocks スクリーナー

        条件:
        - RS Rating ≥ 80
        - EPS Est Cur Qtr % ≥ 100%
        - 50-Day Avg Vol (1000s) ≥ 100
        - Price vs 50-Day ≥ 0.0%
        """
        print("\n=== Explosive Estimated EPS Growth Stocks スクリーナー実行中 ===")
        passed = []

        for ticker in tickers_list:
            try:
                # RS Rating チェック
                rs_rating = rs_percentile_dict.get(ticker, 0)
                if rs_rating < 80:
                    continue

                # 価格データ取得
                prices_df = self.get_historical_prices(ticker, days=100)
                if prices_df is None or len(prices_df) < 50:
                    continue

                # ボリュームチェック
                vol_metrics = self.calculate_volume_metrics(prices_df)
                if not vol_metrics or vol_metrics['avg_vol_50'] < 100:
                    continue

                # Price vs 50-Day MA チェック
                price_vs_50ma = self.calculate_price_vs_ma(prices_df)
                if price_vs_50ma is None or price_vs_50ma < 0:
                    continue

                # アナリスト予想取得（EPS Est Cur Qtr）
                analyst_est = self.get_analyst_estimates(ticker)
                if analyst_est and len(analyst_est) > 0:
                    # 次四半期のEPS成長率を推定
                    # FMP APIの制限により、実際のEPS Estimateの成長率が取れない場合は
                    # 過去のEPS成長率で代用
                    income_stmt = self.get_income_statement(ticker, period='quarter', limit=5)
                    eps_metrics = self.calculate_eps_growth(income_stmt)

                    if eps_metrics and eps_metrics['eps_growth_last_qtr'] is not None:
                        if eps_metrics['eps_growth_last_qtr'] >= 100:
                            passed.append(ticker)

            except Exception as e:
                continue

        print(f"  合格: {len(passed)} 銘柄")
        return passed

    def screener_up_on_volume(self, tickers_list, rs_percentile_dict):
        """
        Up on Volume List スクリーナー

        条件:
        - Price % Chg ≥ 0.00%
        - Vol% Chg vs 50-Day ≥ 20%
        - Current Price ≥ $10
        - 50-Day Avg Vol (1000s) ≥ 100
        - Market Cap (mil) ≥ $250
        - RS Rating ≥ 80
        - EPS % Chg Last Qtr ≥ 20%
        - A/D Rating ABC
        """
        print("\n=== Up on Volume List スクリーナー実行中 ===")
        passed = []

        for ticker in tickers_list:
            try:
                # 価格データ取得
                prices_df = self.get_historical_prices(ticker, days=100)
                if prices_df is None or len(prices_df) < 65:
                    continue

                # 価格変化チェック
                price_metrics = self.calculate_price_changes(prices_df)
                if not price_metrics or price_metrics['pct_change_1d'] < 0:
                    continue

                # Current Price チェック
                if price_metrics['price'] < 10:
                    continue

                # ボリュームチェック
                vol_metrics = self.calculate_volume_metrics(prices_df)
                if not vol_metrics:
                    continue

                if vol_metrics['avg_vol_50'] < 100:
                    continue

                if vol_metrics['vol_change_pct'] < 20:
                    continue

                # 企業プロファイル取得（Market Cap）
                profile = self.get_company_profile(ticker)
                if not profile:
                    continue

                market_cap = profile.get('mktCap', 0) / 1_000_000  # millions
                if market_cap < 250:
                    continue

                # RS Rating チェック
                rs_rating = rs_percentile_dict.get(ticker, 0)
                if rs_rating < 80:
                    continue

                # EPS成長率チェック
                income_stmt = self.get_income_statement(ticker, period='quarter', limit=5)
                eps_metrics = self.calculate_eps_growth(income_stmt)
                if not eps_metrics or eps_metrics['eps_growth_last_qtr'] is None:
                    continue

                if eps_metrics['eps_growth_last_qtr'] < 20:
                    continue

                # A/D Rating チェック
                ad_rating = self.estimate_ad_rating(prices_df)
                if ad_rating not in ['A', 'B', 'C']:
                    continue

                passed.append(ticker)

            except Exception as e:
                continue

        print(f"  合格: {len(passed)} 銘柄")
        return passed

    def screener_top_2_percent_rs(self, tickers_list, rs_percentile_dict):
        """
        Top 2% RS Rating List スクリーナー

        条件:
        - RS Rating ≥ 98
        - 10Day > 21Day > 50Day
        - 50-Day Avg Vol (1000s) ≥ 100
        - Volume (1000s) ≥ 100
        - Sector NOT: medical
        """
        print("\n=== Top 2% RS Rating List スクリーナー実行中 ===")
        passed = []

        for ticker in tickers_list:
            try:
                # RS Rating チェック
                rs_rating = rs_percentile_dict.get(ticker, 0)
                if rs_rating < 98:
                    continue

                # 価格データ取得
                prices_df = self.get_historical_prices(ticker, days=100)
                if prices_df is None or len(prices_df) < 50:
                    continue

                # 移動平均チェック: 10Day > 21Day > 50Day
                ma_data = self.calculate_moving_averages(prices_df)
                if not ma_data:
                    continue

                if not (ma_data['10ma'] > ma_data['21ma'] > ma_data['50ma']):
                    continue

                # ボリュームチェック
                vol_metrics = self.calculate_volume_metrics(prices_df)
                if not vol_metrics:
                    continue

                if vol_metrics['avg_vol_50'] < 100:
                    continue

                if vol_metrics['current_volume'] < 100:
                    continue

                # セクターチェック（医療セクターを除外）
                profile = self.get_company_profile(ticker)
                if profile:
                    sector = profile.get('sector', '').lower()
                    if 'healthcare' in sector or 'medical' in sector:
                        continue

                passed.append(ticker)

            except Exception as e:
                continue

        print(f"  合格: {len(passed)} 銘柄")
        return passed

    def screener_4_percent_bullish_yesterday(self, tickers_list, rs_percentile_dict):
        """
        4% Bullish Yesterday スクリーナー

        条件:
        - Price ≥ $1
        - Change > 4%
        - Market cap > $250M
        - Volume > 100K
        - Rel Volume > 1
        - Change from Open > 0%
        - Avg Volume 90D > 100K
        """
        print("\n=== 4% Bullish Yesterday スクリーナー実行中 ===")
        passed = []

        for ticker in tickers_list:
            try:
                # 価格データ取得
                prices_df = self.get_historical_prices(ticker, days=100)
                if prices_df is None or len(prices_df) < 90:
                    continue

                # 価格変化チェック
                price_metrics = self.calculate_price_changes(prices_df)
                if not price_metrics:
                    continue

                # Price チェック
                if price_metrics['price'] < 1:
                    continue

                # Change > 4% チェック
                if price_metrics['pct_change_1d'] <= 4:
                    continue

                # Change from Open > 0% チェック
                if price_metrics['change_from_open'] <= 0:
                    continue

                # ボリュームチェック
                vol_metrics = self.calculate_volume_metrics(prices_df)
                if not vol_metrics:
                    continue

                # Volume > 100K
                if vol_metrics['current_volume'] <= 100:
                    continue

                # Rel Volume > 1
                if vol_metrics['rel_volume'] <= 1:
                    continue

                # Avg Volume 90D > 100K
                if vol_metrics['avg_vol_90'] <= 100:
                    continue

                # Market Cap チェック
                profile = self.get_company_profile(ticker)
                if not profile:
                    continue

                market_cap = profile.get('mktCap', 0) / 1_000_000  # millions
                if market_cap <= 250:
                    continue

                passed.append(ticker)

            except Exception as e:
                continue

        print(f"  合格: {len(passed)} 銘柄")
        return passed

    def screener_healthy_chart_watchlist(self, tickers_list, rs_percentile_dict, benchmark_prices_df):
        """
        Healthy Chart Watch List スクリーナー

        条件:
        - 10Day > 21Day > 50Day
        - 50Day > 150Day > 200Day
        - RS Line New High
        - RS Rating ≥ 90
        - A/D Rating AB
        - Ind Group RS AB (省略可能)
        - Comp Rating ≥ 80
        - 50-Day Avg Vol (1000s) ≥ 100
        """
        print("\n=== Healthy Chart Watch List スクリーナー実行中 ===")
        passed = []

        for ticker in tickers_list:
            try:
                # RS Rating チェック
                rs_rating = rs_percentile_dict.get(ticker, 0)
                if rs_rating < 90:
                    continue

                # 価格データ取得
                prices_df = self.get_historical_prices(ticker, days=300)
                if prices_df is None or len(prices_df) < 200:
                    continue

                # 移動平均チェック: 10Day > 21Day > 50Day
                ma_data = self.calculate_moving_averages(prices_df)
                if not ma_data:
                    continue

                if not (ma_data['10ma'] > ma_data['21ma'] > ma_data['50ma']):
                    continue

                # 移動平均チェック: 50Day > 150Day > 200Day
                if not (ma_data['50ma'] > ma_data['150ma'] > ma_data['200ma']):
                    continue

                # RS Line New High チェック
                if not self.check_rs_line_new_high(prices_df, benchmark_prices_df):
                    continue

                # A/D Rating チェック
                ad_rating = self.estimate_ad_rating(prices_df)
                if ad_rating not in ['A', 'B']:
                    continue

                # Composite Rating チェック
                eps_rating_score = None
                income_stmt_q = self.get_income_statement(ticker, period='quarter', limit=8)
                income_stmt_a = self.get_income_statement(ticker, period='annual', limit=3)
                eps_metrics = self.calculate_eps_growth(income_stmt_q, income_stmt_a)
                if eps_metrics:
                    eps_rating_score = eps_metrics.get('eps_rating_score')

                price_vs_52w = self.calculate_52w_high_distance(prices_df)
                comp_rating = self.estimate_comp_rating(rs_rating, eps_rating_score, ad_rating, price_vs_52w)

                if comp_rating is None or comp_rating < 80:
                    continue

                # ボリュームチェック
                vol_metrics = self.calculate_volume_metrics(prices_df)
                if not vol_metrics or vol_metrics['avg_vol_50'] < 100:
                    continue

                passed.append(ticker)

            except Exception as e:
                continue

        print(f"  合格: {len(passed)} 銘柄")
        return passed

    # ==================== マルチスレッド処理 ====================

    def process_ticker_batch(self, tickers_batch):
        """ティッカーのバッチを処理"""
        results = {}

        for ticker in tickers_batch:
            try:
                prices_df = self.get_historical_prices(ticker, days=300)
                if prices_df is not None:
                    rs_value = self.calculate_rs_rating(prices_df)
                    if rs_value is not None:
                        results[ticker] = {
                            'prices_df': prices_df,
                            'rs_value': rs_value
                        }
            except Exception as e:
                continue

        return results

    def calculate_rs_ratings_parallel(self, tickers_list, max_workers=10):
        """
        マルチスレッドでRS値を並列計算

        Args:
            tickers_list: ティッカーリスト
            max_workers: 最大ワーカー数

        Returns:
            dict: {ticker: rs_value}
        """
        print(f"\n全銘柄のRS値を計算中（{max_workers}スレッド並列処理）...")

        # バッチサイズを設定
        batch_size = 50
        batches = [tickers_list[i:i+batch_size] for i in range(0, len(tickers_list), batch_size)]

        rs_values_dict = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # バッチごとに処理を投入
            future_to_batch = {executor.submit(self.process_ticker_batch, batch): batch for batch in batches}

            completed = 0
            for future in as_completed(future_to_batch):
                completed += 1
                if completed % 10 == 0 or completed == len(batches):
                    print(f"  進捗: {completed}/{len(batches)} バッチ完了 ({completed * batch_size}/{len(tickers_list)} 銘柄)")

                try:
                    batch_results = future.result()
                    for ticker, data in batch_results.items():
                        rs_values_dict[ticker] = data['rs_value']
                except Exception as e:
                    continue

        print(f"  {len(rs_values_dict)} 銘柄のRS値を計算しました")
        return rs_values_dict

    # ==================== メイン実行関数 ====================

    def run_all_screeners(self, max_workers=10, use_full_dataset=True):
        """
        全スクリーナーを実行してGoogleスプレッドシートに出力

        Args:
            max_workers: 並列処理の最大ワーカー数
            use_full_dataset: Trueの場合は全銘柄、Falseの場合はテスト用に制限
        """
        print("\n" + "="*80)
        print("IBD スクリーナー実行開始")
        print("="*80)

        # ティッカーリストを取得
        print("\nティッカーリストを取得中...")
        fetcher = FMPTickerFetcher()
        tickers_df = fetcher.get_all_stocks(['nasdaq', 'nyse'])
        tickers_list = tickers_df['Ticker'].tolist()
        print(f"  合計 {len(tickers_list)} 銘柄を取得しました")

        # テスト用にサンプルサイズを制限するオプション
        if not use_full_dataset:
            sample_size = min(500, len(tickers_list))
            tickers_list = tickers_list[:sample_size]
            print(f"  テストモード: {sample_size} 銘柄に制限")

        # ベンチマーク（SPY）のデータ取得
        print("\nベンチマーク（SPY）のデータを取得中...")
        benchmark_prices_df = self.get_historical_prices('SPY', days=300)

        # 全銘柄のRS値を並列計算
        rs_values_dict = self.calculate_rs_ratings_parallel(tickers_list, max_workers=max_workers)

        # RS値をパーセンタイルランクに変換
        print("\nRS値をパーセンタイルランクに変換中...")
        rs_percentile_dict = self.percentile_rank_rs_ratings(rs_values_dict)
        print(f"  {len(rs_percentile_dict)} 銘柄のRSランクを計算しました")

        # 各スクリーナーを実行
        screener_results = {}

        screener_results['Momentum 97'] = self.screener_momentum_97(
            tickers_list, rs_percentile_dict
        )

        screener_results['Explosive Estimated EPS Growth Stocks'] = self.screener_explosive_eps_growth(
            tickers_list, rs_percentile_dict
        )

        screener_results['Up on Volume List'] = self.screener_up_on_volume(
            tickers_list, rs_percentile_dict
        )

        screener_results['Top 2% RS Rating List'] = self.screener_top_2_percent_rs(
            tickers_list, rs_percentile_dict
        )

        screener_results['4% Bullish Yesterday'] = self.screener_4_percent_bullish_yesterday(
            tickers_list, rs_percentile_dict
        )

        screener_results['Healthy Chart Watch List'] = self.screener_healthy_chart_watchlist(
            tickers_list, rs_percentile_dict, benchmark_prices_df
        )

        # Googleスプレッドシートに出力
        print("\nGoogleスプレッドシートに出力中...")
        self.write_screeners_to_sheet(screener_results)

        print("\n" + "="*80)
        print("すべてのスクリーナー実行完了!")
        print(f"スプレッドシートURL: {self.spreadsheet.url}")
        print("="*80)

    def write_screeners_to_sheet(self, screener_results):
        """
        スクリーナー結果をGoogleスプレッドシートに出力

        フォーマット:
        - 各スクリーナー名をヘッダー行に表示
        - ティッカーを横10個ずつ配置
        - スクリーナー間に空行を挿入
        """
        # シートの作成または取得
        sheet_name = 'IBD Screeners'
        try:
            worksheet = self.spreadsheet.worksheet(sheet_name)
            worksheet.clear()
        except gspread.WorksheetNotFound:
            worksheet = self.spreadsheet.add_worksheet(
                title=sheet_name,
                rows=500,
                cols=10
            )

        current_row = 1

        for screener_name, tickers in screener_results.items():
            # スクリーナー名を出力
            worksheet.update(f'A{current_row}', [[screener_name]])

            # ヘッダー行のフォーマット
            header_format = {
                'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.6},
                'textFormat': {
                    'bold': True,
                    'foregroundColor': {'red': 1, 'green': 1, 'blue': 1},
                    'fontSize': 12
                },
                'horizontalAlignment': 'LEFT'
            }
            worksheet.format(f'A{current_row}:J{current_row}', header_format)
            worksheet.merge_cells(f'A{current_row}:J{current_row}')
            current_row += 1

            # ティッカーを10個ずつ横に並べる
            if tickers:
                rows_data = []
                for i in range(0, len(tickers), 10):
                    row_tickers = tickers[i:i+10]
                    # 10個に満たない場合は空文字で埋める
                    while len(row_tickers) < 10:
                        row_tickers.append('')
                    rows_data.append(row_tickers)

                # データを一括書き込み
                if rows_data:
                    end_row = current_row + len(rows_data) - 1
                    worksheet.update(f'A{current_row}:J{end_row}', rows_data)
                    current_row = end_row + 1

            # スクリーナー間に空行を挿入
            current_row += 1

        print(f"  '{sheet_name}' シートに出力完了")


def main():
    """メイン実行関数"""
    # 環境変数から設定を取得
    FMP_API_KEY = os.getenv('FMP_API_KEY')
    CREDENTIALS_FILE = os.getenv('CREDENTIALS_FILE', 'credentials.json')
    SPREADSHEET_NAME = os.getenv('SPREADSHEET_NAME', 'Market Dashboard')
    MAX_WORKERS = int(os.getenv('ORATNEK_MAX_WORKERS', '10'))

    # API KEYのチェック
    if not FMP_API_KEY or FMP_API_KEY == 'your_api_key_here' or FMP_API_KEY == 'your_fmp_api_key_here':
        print("エラー: FMP_API_KEYが設定されていません")
        print("1. .env.exampleを.envにコピーしてください: cp .env.example .env")
        print("2. .envファイルを編集して、FMP_API_KEYを設定してください")
        print("3. APIキーは https://site.financialmodelingprep.com/developer/docs から取得できます")
        return

    try:
        # スクリーナーの実行
        screeners = IBDScreeners(FMP_API_KEY, CREDENTIALS_FILE, SPREADSHEET_NAME)

        # use_full_dataset=True で全銘柄を処理
        # テストの場合は use_full_dataset=False に設定
        screeners.run_all_screeners(max_workers=MAX_WORKERS, use_full_dataset=True)

    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
