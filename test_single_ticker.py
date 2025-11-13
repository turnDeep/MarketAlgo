"""
単一銘柄でAPIテスト
"""
import os
from dotenv import load_dotenv
from ibd_data_collector import IBDDataCollector

load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY')

print(f"API Key: {FMP_API_KEY}")
print(f"API Key length: {len(FMP_API_KEY) if FMP_API_KEY else 0}")
print()

if not FMP_API_KEY or FMP_API_KEY == 'demo':
    print("警告: APIキーが'demo'または未設定です")
    print("FMP API は 'demo' キーを受け付けなくなっています")
    print()
    print("解決方法:")
    print("1. https://site.financialmodelingprep.com/developer/docs/ にアクセス")
    print("2. 無料アカウントを作成してAPIキーを取得")
    print("3. .envファイルにAPIキーを設定:")
    print("   FMP_API_KEY=your_actual_api_key_here")
    print()

    # demoキーでも続行してエラーを確認
    print("demoキーでテストを続行してエラーを確認します...")
    print()

collector = IBDDataCollector(FMP_API_KEY)

print("="*80)
print("テスト1: 株価データ取得 (AAPL)")
print("="*80)
prices = collector.get_historical_prices("AAPL", days=300)
if prices is not None:
    print(f"✓ 成功: {len(prices)} 日分のデータを取得")
    print(f"  最新日付: {prices['date'].iloc[-1]}")
    print(f"  最新価格: ${prices['close'].iloc[-1]:.2f}")
else:
    print("✗ 失敗: データを取得できませんでした")

print()
print("="*80)
print("テスト2: 損益計算書取得 (AAPL)")
print("="*80)
income = collector.get_income_statement("AAPL", period='quarter', limit=8)
if income:
    print(f"✓ 成功: {len(income)} 四半期分のデータを取得")
    print(f"  最新四半期: {income[0].get('date', 'N/A')}")
    print(f"  EPS: ${income[0].get('eps', 'N/A')}")
else:
    print("✗ 失敗: データを取得できませんでした")

print()
print("="*80)
print("テスト3: 企業プロファイル取得 (AAPL)")
print("="*80)
profile = collector.get_company_profile("AAPL")
if profile:
    print(f"✓ 成功: プロファイルを取得")
    print(f"  会社名: {profile.get('companyName', 'N/A')}")
    print(f"  セクター: {profile.get('sector', 'N/A')}")
else:
    print("✗ 失敗: データを取得できませんでした")

collector.close()

print()
print("="*80)
print("テスト完了")
print("="*80)
