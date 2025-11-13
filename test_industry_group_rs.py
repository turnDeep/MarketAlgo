"""
Industry Group RS実装のテストスクリプト

このスクリプトは、Industry Group RSの実装をテストします。
"""

import os
from dotenv import load_dotenv
from ibd_data_collector import IBDDataCollector
from ibd_ratings_calculator import IBDRatingsCalculator

def test_industry_group_rs():
    """Industry Group RS実装のテスト"""

    load_dotenv()

    FMP_API_KEY = os.getenv('FMP_API_KEY')
    if not FMP_API_KEY or FMP_API_KEY == 'your_api_key_here':
        print("エラー: FMP_API_KEYが設定されていません")
        return

    print("\n" + "="*80)
    print("Industry Group RS実装テスト")
    print("="*80)

    # 1. データコレクターの初期化
    print("\n1. データコレクターの初期化...")
    collector = IBDDataCollector(FMP_API_KEY)

    # 2. セクターパフォーマンスデータの収集（既に存在する場合はスキップ）
    print("\n2. セクターパフォーマンスデータの収集...")
    try:
        collector.collect_sector_performance_data(limit=300)
        print("   セクターパフォーマンスデータ収集完了")
    except Exception as e:
        print(f"   エラー: {str(e)}")

    collector.close()

    # 3. レーティング計算機の初期化
    print("\n3. レーティング計算機の初期化...")
    calculator = IBDRatingsCalculator()

    # 4. テスト用の銘柄でIndustry Group RSを計算
    print("\n4. サンプル銘柄でIndustry Group RSをテスト...")
    test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    for ticker in test_tickers:
        try:
            ig_rs_data = calculator.calculate_industry_group_rs(ticker)
            if ig_rs_data:
                print(f"\n   {ticker}:")
                print(f"     セクター: {ig_rs_data['sector']}")
                print(f"     業界: {ig_rs_data['industry']}")
                print(f"     株式RS値: {ig_rs_data['stock_rs_value']:.2f}")
                print(f"     セクターRS値: {ig_rs_data['sector_rs_value']:.2f}")
                print(f"     Industry Group RS: {ig_rs_data['industry_group_rs_value']:.2f}")
            else:
                print(f"\n   {ticker}: データなし（セクターデータまたは株価データが不足）")
        except Exception as e:
            print(f"\n   {ticker}: エラー - {str(e)}")

    # 5. データベース統計を表示
    print("\n5. データベース統計:")
    calculator.db.get_database_stats()

    calculator.close()

    print("\n" + "="*80)
    print("テスト完了")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_industry_group_rs()
