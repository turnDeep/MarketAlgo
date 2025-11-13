"""
単一ティッカーでFMP APIをテストするスクリプト
エラーメッセージを詳細に表示して問題を特定します
"""

import os
import sys
from dotenv import load_dotenv
from ibd_data_collector import IBDDataCollector

# 環境変数を読み込む
load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY')

print("="*80)
print("単一ティッカーテスト")
print("="*80)

# API KEYのチェック
if not FMP_API_KEY or FMP_API_KEY == 'your_api_key_here' or FMP_API_KEY == 'your_fmp_api_key_here':
    print("\nエラー: FMP_API_KEYが設定されていません")
    print("\n対処方法:")
    print("1. .env.exampleを.envにコピーしてください:")
    print("   cp .env.example .env")
    print("\n2. .envファイルを編集して、FMP_API_KEYを設定してください:")
    print("   FMP_API_KEY=あなたのAPIキー")
    print("\n3. APIキーは https://financialmodelingprep.com/developer/docs から取得できます")
    sys.exit(1)

print(f"\nAPIキー: ...{FMP_API_KEY[-8:]} (長さ: {len(FMP_API_KEY)}文字)")

# テスト用ティッカー
test_tickers = ['AAPL', 'MSFT', 'GOOGL']

print(f"\nテストティッカー: {', '.join(test_tickers)}")
print("="*80)

try:
    # データコレクターを初期化
    collector = IBDDataCollector(FMP_API_KEY, db_path='test_ibd.db')

    for ticker in test_tickers:
        print(f"\n[{ticker}] データ収集を開始...")
        print("-"*80)

        success = collector.collect_ticker_data(ticker)

        if success:
            print(f"✓ [{ticker}] データ収集成功！")
        else:
            print(f"✗ [{ticker}] データ収集失敗")

        print("-"*80)

    collector.close()

    print("\n" + "="*80)
    print("テスト完了")
    print("="*80)

except Exception as e:
    print(f"\n✗ エラーが発生しました:")
    print(f"  {type(e).__name__}: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
