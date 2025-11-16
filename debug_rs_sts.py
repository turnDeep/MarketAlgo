"""
RS STS%計算のデバッグスクリプト

データベースの状態を調査し、RS STS%がNoneになる原因を特定します。
"""

import pandas as pd
from ibd_database import IBDDatabase


def debug_rs_sts_calculation():
    """RS STS%計算のデバッグ"""
    db = IBDDatabase('ibd_data.db')

    print("="*80)
    print("RS STS% 計算デバッグ")
    print("="*80)

    # 1. データベース全体の統計を確認
    print("\n1. データベース統計:")
    db.get_database_stats()

    # 2. サンプル銘柄を選択（最初の10銘柄）
    all_tickers = db.get_all_tickers()
    print(f"\n2. 全銘柄数: {len(all_tickers)}")

    if len(all_tickers) == 0:
        print("エラー: データベースに銘柄が存在しません")
        db.close()
        return

    sample_tickers = all_tickers[:10]
    print(f"   サンプル銘柄: {sample_tickers}")

    # 3. ベンチマーク（SPY）の価格データを確認
    print("\n3. ベンチマーク (SPY) の価格データ:")
    spy_prices = db.get_price_history('SPY', days=30)
    if spy_prices is not None:
        print(f"   SPY 価格データ件数: {len(spy_prices)} 日分")
        print(f"   最新日: {spy_prices['date'].max()}")
        print(f"   最古日: {spy_prices['date'].min()}")
        print("\n   最新5日分:")
        print(spy_prices.tail(5))
    else:
        print("   SPY 価格データが見つかりません！")

    # 4. 各サンプル銘柄の価格データを確認
    print("\n4. サンプル銘柄の価格データ:")
    for ticker in sample_tickers:
        ticker_prices = db.get_price_history(ticker, days=30)
        if ticker_prices is not None:
            print(f"   {ticker}: {len(ticker_prices)} 日分 (最新: {ticker_prices['date'].max()})")
        else:
            print(f"   {ticker}: データなし")

    # 5. 日付マージのテスト（最初のサンプル銘柄）
    if len(sample_tickers) > 0:
        test_ticker = sample_tickers[0]
        print(f"\n5. 日付マージテスト ({test_ticker} vs SPY):")

        if spy_prices is not None:
            ticker_prices = db.get_price_history(test_ticker, days=30)

            if ticker_prices is not None:
                # 日付マージ
                merged = pd.merge(
                    spy_prices[['date', 'close']].rename(columns={'close': 'spy_close'}),
                    ticker_prices[['date', 'close']].rename(columns={'close': 'ticker_close'}),
                    on='date',
                    how='inner'
                )

                print(f"   SPY データ: {len(spy_prices)} 日分")
                print(f"   {test_ticker} データ: {len(ticker_prices)} 日分")
                print(f"   マージ後: {len(merged)} 日分")

                if len(merged) >= 25:
                    print(f"   ✓ 十分なデータあり (25日以上)")
                else:
                    print(f"   ✗ データ不足 (25日未満)")
                    print(f"\n   SPY の日付範囲:")
                    print(f"     最古: {spy_prices['date'].min()}")
                    print(f"     最新: {spy_prices['date'].max()}")
                    print(f"\n   {test_ticker} の日付範囲:")
                    print(f"     最古: {ticker_prices['date'].min()}")
                    print(f"     最新: {ticker_prices['date'].max()}")

                    # 共通する日付がない原因を調査
                    spy_dates = set(spy_prices['date'].dt.date)
                    ticker_dates = set(ticker_prices['date'].dt.date)
                    common_dates = spy_dates & ticker_dates
                    print(f"\n   共通する日付: {len(common_dates)} 日")
                    if len(common_dates) > 0:
                        print(f"   共通日付の例: {sorted(list(common_dates))[:5]}")
            else:
                print(f"   {test_ticker} のデータが取得できません")
        else:
            print("   SPYのデータが取得できません")

    # 6. RS値の計算状態を確認
    print("\n6. RS値の計算状態:")
    rs_values = db.get_all_rs_values()
    print(f"   RS値が計算済みの銘柄数: {len(rs_values)}")

    if len(rs_values) > 0:
        # サンプル銘柄のRS値を表示
        print(f"   サンプル銘柄のRS値:")
        for ticker in sample_tickers[:5]:
            rs_value = rs_values.get(ticker)
            print(f"     {ticker}: {rs_value}")

    db.close()

    print("\n" + "="*80)
    print("デバッグ完了")
    print("="*80)


if __name__ == "__main__":
    debug_rs_sts_calculation()
