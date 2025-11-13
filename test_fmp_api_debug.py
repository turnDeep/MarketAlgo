"""
FMP API デバッグスクリプト
実際のエラーメッセージを確認するために使用
"""

import os
from dotenv import load_dotenv
from curl_cffi.requests import Session

# 環境変数を読み込む
load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY')

print("="*80)
print("FMP API デバッグテスト")
print("="*80)
print(f"\nAPIキーの確認:")
if FMP_API_KEY:
    print(f"  APIキーが設定されています（長さ: {len(FMP_API_KEY)}文字）")
    print(f"  最初の5文字: {FMP_API_KEY[:5]}...")
else:
    print("  エラー: APIキーが設定されていません")
    exit(1)

# テスト用のティッカー
test_ticker = "AAPL"

# curl_cffiセッションを初期化
print("\ncurl_cffiセッションを初期化中...")
session = Session(impersonate="chrome110")

# テスト1: 株価データの取得
print(f"\n{'='*80}")
print(f"テスト1: 株価データの取得（{test_ticker}）")
print(f"{'='*80}")
url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{test_ticker}"
params = {
    'apikey': FMP_API_KEY,
    'timeseries': 300
}

try:
    print(f"\nリクエストURL: {url}")
    print(f"パラメータ: timeseries=300, apikey=...{FMP_API_KEY[-5:]}")

    response = session.get(url, params=params, timeout=30)
    print(f"\nレスポンスステータスコード: {response.status_code}")
    print(f"レスポンスヘッダー:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ 成功！")
        if 'historical' in data:
            print(f"  取得したデータポイント数: {len(data['historical'])}件")
            print(f"  最新のデータ: {data['historical'][0] if data['historical'] else 'なし'}")
        else:
            print(f"  レスポンス内容: {data}")
    else:
        print(f"\n✗ 失敗！")
        print(f"レスポンステキスト: {response.text[:500]}")

except Exception as e:
    print(f"\n✗ 例外が発生しました:")
    print(f"  エラータイプ: {type(e).__name__}")
    print(f"  エラーメッセージ: {str(e)}")
    import traceback
    traceback.print_exc()

# テスト2: 企業プロファイルの取得
print(f"\n{'='*80}")
print(f"テスト2: 企業プロファイルの取得（{test_ticker}）")
print(f"{'='*80}")
url = f"https://financialmodelingprep.com/api/v3/profile/{test_ticker}"
params = {'apikey': FMP_API_KEY}

try:
    print(f"\nリクエストURL: {url}")

    response = session.get(url, params=params, timeout=30)
    print(f"\nレスポンスステータスコード: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ 成功！")
        if data and len(data) > 0:
            profile = data[0]
            print(f"  会社名: {profile.get('companyName', 'N/A')}")
            print(f"  セクター: {profile.get('sector', 'N/A')}")
            print(f"  時価総額: {profile.get('mktCap', 'N/A')}")
        else:
            print(f"  レスポンス内容: {data}")
    else:
        print(f"\n✗ 失敗！")
        print(f"レスポンステキスト: {response.text[:500]}")

except Exception as e:
    print(f"\n✗ 例外が発生しました:")
    print(f"  エラータイプ: {type(e).__name__}")
    print(f"  エラーメッセージ: {str(e)}")
    import traceback
    traceback.print_exc()

# テスト3: 損益計算書の取得
print(f"\n{'='*80}")
print(f"テスト3: 損益計算書の取得（{test_ticker}）")
print(f"{'='*80}")
url = f"https://financialmodelingprep.com/api/v3/income-statement/{test_ticker}"
params = {
    'apikey': FMP_API_KEY,
    'period': 'quarter',
    'limit': 8
}

try:
    print(f"\nリクエストURL: {url}")
    print(f"パラメータ: period=quarter, limit=8")

    response = session.get(url, params=params, timeout=30)
    print(f"\nレスポンスステータスコード: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ 成功！")
        if data and len(data) > 0:
            print(f"  取得した四半期数: {len(data)}件")
            print(f"  最新四半期: {data[0].get('date', 'N/A')}")
            print(f"  EPS: {data[0].get('eps', 'N/A')}")
        else:
            print(f"  レスポンス内容: {data}")
    else:
        print(f"\n✗ 失敗！")
        print(f"レスポンステキスト: {response.text[:500]}")

except Exception as e:
    print(f"\n✗ 例外が発生しました:")
    print(f"  エラータイプ: {type(e).__name__}")
    print(f"  エラーメッセージ: {str(e)}")
    import traceback
    traceback.print_exc()

# セッションをクローズ
session.close()

print(f"\n{'='*80}")
print("デバッグテスト完了")
print(f"{'='*80}\n")
