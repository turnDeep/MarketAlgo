"""
IBD Ratings Calculator

データベースに保存された生の値（RS値、EPS要素）をパーセンタイルランキングに変換し、
最終的なレーティング（RS Rating, EPS Rating, Composite Rating）を計算します。

オプション2: 正確な実装
- RS Ratingと同様に、全銘柄のEPSデータを収集
- 各要素（最新四半期成長率、前四半期成長率、年間成長率、安定性）を個別にパーセンタイルランキング
- 重み付けして最終EPSランキングを計算
"""

import numpy as np
from typing import Dict, List, Optional
from ibd_database import IBDDatabase


class IBDRatingsCalculator:
    """IBDレーティング計算クラス"""

    def __init__(self, db_path: str = 'ibd_data.db'):
        """
        Args:
            db_path: データベースファイルのパス
        """
        self.db = IBDDatabase(db_path)

    def close(self):
        """リソースをクリーンアップ"""
        self.db.close()

    # ==================== パーセンタイルランキング計算 ====================

    def calculate_percentile_ranking(self, values_dict: Dict[str, float]) -> Dict[str, float]:
        """
        値の辞書をパーセンタイルランキング（0-100）に変換

        Args:
            values_dict: {ticker: value} の辞書

        Returns:
            dict: {ticker: percentile_rating} の辞書（0-100）
        """
        # Noneや無効な値を除外
        valid_values = {k: v for k, v in values_dict.items() if v is not None and not np.isnan(v)}

        if not valid_values:
            return {}

        # 値でソート
        sorted_tickers = sorted(valid_values.items(), key=lambda x: x[1])

        # パーセンタイルを計算
        percentile_dict = {}
        total = len(sorted_tickers)

        for idx, (ticker, val) in enumerate(sorted_tickers):
            # パーセンタイル: (順位 / 総数) * 100
            percentile = ((idx + 1) / total) * 100
            percentile_dict[ticker] = round(percentile, 2)

        return percentile_dict

    # ==================== RS Rating計算 ====================

    def calculate_rs_ratings(self) -> Dict[str, float]:
        """
        全銘柄のRS Ratingを計算（パーセンタイルランキング方式）

        1. データベースから全銘柄のRS値を取得
        2. RS値をパーセンタイルランキング（0-100）に変換
        3. データベースに保存

        Returns:
            dict: {ticker: rs_rating} の辞書
        """
        print("\n全銘柄のRS Ratingを計算中...")

        # 1. 全銘柄のRS値を取得
        rs_values_dict = self.db.get_all_rs_values()
        print(f"  {len(rs_values_dict)} 銘柄のRS値を取得")

        # 2. パーセンタイルランキングに変換
        rs_ratings_dict = self.calculate_percentile_ranking(rs_values_dict)
        print(f"  {len(rs_ratings_dict)} 銘柄のRS Ratingを計算")

        # 3. データベースに保存（後で一括更新）
        return rs_ratings_dict

    # ==================== EPS Rating計算（オプション2: 正確な実装） ====================

    def calculate_eps_ratings(self) -> Dict[str, float]:
        """
        全銘柄のEPS Ratingを計算（パーセンタイルランキング方式）

        オプション2の実装:
        1. データベースから全銘柄のEPS要素を取得
        2. 各要素を個別にパーセンタイルランキング（0-100）に変換
           - 最新四半期EPS成長率
           - 前四半期EPS成長率
           - 年間EPS成長率（CAGR）
           - 収益安定性スコア
        3. 重み付けして最終EPSランキングを計算
           - 最新四半期: 50%
           - 前四半期: 20%
           - 年間成長率: 20%
           - 安定性: 10%

        Returns:
            dict: {ticker: eps_rating} の辞書
        """
        print("\n全銘柄のEPS Ratingを計算中（パーセンタイルランキング方式）...")

        # 1. 全銘柄のEPS要素を取得
        eps_components_dict = self.db.get_all_eps_components()
        print(f"  {len(eps_components_dict)} 銘柄のEPS要素を取得")

        # 2. 各要素を個別に抽出
        eps_growth_last_qtr_dict = {}
        eps_growth_prev_qtr_dict = {}
        annual_growth_rate_dict = {}
        stability_score_dict = {}

        for ticker, components in eps_components_dict.items():
            if components['eps_growth_last_qtr'] is not None:
                eps_growth_last_qtr_dict[ticker] = components['eps_growth_last_qtr']
            if components['eps_growth_prev_qtr'] is not None:
                eps_growth_prev_qtr_dict[ticker] = components['eps_growth_prev_qtr']
            if components['annual_growth_rate'] is not None:
                annual_growth_rate_dict[ticker] = components['annual_growth_rate']
            if components['stability_score'] is not None:
                stability_score_dict[ticker] = components['stability_score']

        print(f"    最新四半期成長率: {len(eps_growth_last_qtr_dict)} 銘柄")
        print(f"    前四半期成長率: {len(eps_growth_prev_qtr_dict)} 銘柄")
        print(f"    年間成長率: {len(annual_growth_rate_dict)} 銘柄")
        print(f"    安定性スコア: {len(stability_score_dict)} 銘柄")

        # 3. 各要素を個別にパーセンタイルランキングに変換
        print("  各要素をパーセンタイルランキングに変換中...")

        percentile_last_qtr = self.calculate_percentile_ranking(eps_growth_last_qtr_dict)
        percentile_prev_qtr = self.calculate_percentile_ranking(eps_growth_prev_qtr_dict)
        percentile_annual = self.calculate_percentile_ranking(annual_growth_rate_dict)
        percentile_stability = self.calculate_percentile_ranking(stability_score_dict)

        # 4. 重み付けして最終EPSランキングを計算
        print("  重み付けして最終EPSランキングを計算中...")

        # IBDの重み付け:
        # - 最新四半期: 50%
        # - 前四半期: 20%
        # - 年間成長率: 20%
        # - 安定性: 10%
        weights = {
            'last_qtr': 0.50,
            'prev_qtr': 0.20,
            'annual': 0.20,
            'stability': 0.10
        }

        eps_ratings_dict = {}

        # 全ティッカーのユニオンを取得
        all_tickers = set()
        all_tickers.update(percentile_last_qtr.keys())
        all_tickers.update(percentile_prev_qtr.keys())
        all_tickers.update(percentile_annual.keys())
        all_tickers.update(percentile_stability.keys())

        for ticker in all_tickers:
            total_score = 0
            total_weight = 0

            # 最新四半期（50%）
            if ticker in percentile_last_qtr:
                total_score += percentile_last_qtr[ticker] * weights['last_qtr']
                total_weight += weights['last_qtr']

            # 前四半期（20%）
            if ticker in percentile_prev_qtr:
                total_score += percentile_prev_qtr[ticker] * weights['prev_qtr']
                total_weight += weights['prev_qtr']

            # 年間成長率（20%）
            if ticker in percentile_annual:
                total_score += percentile_annual[ticker] * weights['annual']
                total_weight += weights['annual']

            # 安定性（10%）
            if ticker in percentile_stability:
                total_score += percentile_stability[ticker] * weights['stability']
                total_weight += weights['stability']

            # 重み付けを正規化
            if total_weight > 0:
                eps_rating = total_score / total_weight
                eps_ratings_dict[ticker] = round(eps_rating, 2)

        print(f"  {len(eps_ratings_dict)} 銘柄のEPS Ratingを計算")

        return eps_ratings_dict

    # ==================== A/D Rating計算 ====================

    def calculate_ad_rating(self, ticker: str) -> Optional[str]:
        """
        A/D Rating（Accumulation/Distribution）を計算

        Args:
            ticker: ティッカーシンボル

        Returns:
            str: A, B, C, D, E のレーティング
        """
        prices_df = self.db.get_price_history(ticker, days=70)
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
            high_low_diff = high - low
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

                # レーティング判定
                if combined_score >= 0.4:
                    return 'A'
                elif combined_score >= 0.15:
                    return 'B'
                elif combined_score >= -0.15:
                    return 'C'
                elif combined_score >= -0.4:
                    return 'D'
                else:
                    return 'E'

            return 'C'

        except Exception as e:
            return None

    # ==================== 52週高値からの距離計算 ====================

    def calculate_52w_high_distance(self, ticker: str) -> Optional[float]:
        """52週高値からの距離を計算"""
        prices_df = self.db.get_price_history(ticker, days=252)
        if prices_df is None or len(prices_df) < 252:
            return None

        try:
            recent_data = prices_df.tail(252)
            high_52w = recent_data['high'].max()
            current_price = prices_df['close'].iloc[-1]

            distance = ((current_price - high_52w) / high_52w) * 100
            return round(distance, 2)
        except Exception as e:
            return None

    # ==================== Composite Rating計算 ====================

    def calculate_comp_rating(self, rs_rating: float, eps_rating: float, ad_rating: str,
                             price_vs_52w_high: float) -> Optional[float]:
        """
        Composite Rating を計算

        重み配分:
        - RS Rating: 30% (ダブルウェイト)
        - EPS Rating: 30% (ダブルウェイト)
        - A/D Rating: 15%
        - 52週高値からの距離: 5%
        - Industry Group RS: 10% (未実装)
        - SMR Rating: 10% (未実装)

        Args:
            rs_rating: RS Rating (0-100)
            eps_rating: EPS Rating (0-100)
            ad_rating: A/D Rating (A-E)
            price_vs_52w_high: 52週高値からの距離 (%)

        Returns:
            float: Composite Rating (0-100)
        """
        try:
            score = 0
            weight_sum = 0

            # 1. RS Rating (30% - ダブルウェイト)
            if rs_rating is not None:
                score += rs_rating * 0.30
                weight_sum += 0.30

            # 2. EPS Rating (30% - ダブルウェイト)
            if eps_rating is not None:
                score += eps_rating * 0.30
                weight_sum += 0.30

            # 3. A/D Rating (15%)
            ad_score_map = {
                'A': 100,
                'B': 75,
                'C': 50,
                'D': 25,
                'E': 0
            }
            if ad_rating and ad_rating in ad_score_map:
                score += ad_score_map[ad_rating] * 0.15
                weight_sum += 0.15

            # 4. 52週高値からの距離 (5%)
            if price_vs_52w_high is not None:
                if price_vs_52w_high >= -5:
                    high_score = 100
                elif price_vs_52w_high >= -15:
                    high_score = 100 - ((abs(price_vs_52w_high) - 5) * 5)
                else:
                    high_score = max(0, 50 - ((abs(price_vs_52w_high) - 15) * 3.33))

                score += high_score * 0.05
                weight_sum += 0.05

            # 重み付けを正規化
            if weight_sum > 0:
                normalized_score = score / weight_sum * 100
                return round(min(100, max(0, normalized_score)), 2)
            else:
                return None

        except Exception as e:
            return None

    # ==================== 全レーティングの計算と保存 ====================

    def calculate_all_ratings(self):
        """
        全銘柄のレーティングを計算してデータベースに保存

        1. RS Ratingを計算
        2. EPS Ratingを計算（パーセンタイルランキング方式）
        3. 各銘柄のA/D Rating、52W High Distance、Composite Ratingを計算
        4. データベースに保存
        """
        print(f"\n{'='*80}")
        print("全レーティング計算開始")
        print(f"{'='*80}")

        # 1. RS Ratingを計算
        rs_ratings_dict = self.calculate_rs_ratings()

        # 2. EPS Ratingを計算（パーセンタイルランキング方式）
        eps_ratings_dict = self.calculate_eps_ratings()

        # 3. 全銘柄のティッカーリストを取得
        all_tickers = self.db.get_all_tickers()
        print(f"\n全 {len(all_tickers)} 銘柄のA/D RatingとComposite Ratingを計算中...")

        # 4. 各銘柄のA/D Rating、52W High Distance、Composite Ratingを計算
        calculated_count = 0
        for idx, ticker in enumerate(all_tickers):
            if (idx + 1) % 500 == 0:
                print(f"  進捗: {idx + 1}/{len(all_tickers)} 銘柄")

            try:
                rs_rating = rs_ratings_dict.get(ticker)
                eps_rating = eps_ratings_dict.get(ticker)

                # RS RatingまたはEPS Ratingがない場合はスキップ
                if rs_rating is None and eps_rating is None:
                    continue

                # A/D Ratingを計算
                ad_rating = self.calculate_ad_rating(ticker)

                # 52週高値からの距離を計算
                price_vs_52w_high = self.calculate_52w_high_distance(ticker)

                # Composite Ratingを計算
                comp_rating = self.calculate_comp_rating(rs_rating, eps_rating, ad_rating, price_vs_52w_high)

                # データベースに保存
                self.db.insert_calculated_rating(
                    ticker,
                    rs_rating,
                    eps_rating,
                    ad_rating,
                    comp_rating,
                    price_vs_52w_high
                )

                calculated_count += 1

            except Exception as e:
                continue

        print(f"  {calculated_count} 銘柄のレーティングを計算・保存しました")

        print(f"\n{'='*80}")
        print("全レーティング計算完了!")
        print(f"{'='*80}\n")

        # 統計表示
        self.db.get_database_stats()


def main():
    """テスト実行"""
    try:
        calculator = IBDRatingsCalculator()

        # 全レーティングを計算
        calculator.calculate_all_ratings()

        calculator.close()

    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
