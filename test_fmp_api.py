"""
FMP API テストスクリプト
実際のAPIレスポンスとエラーを確認
requests と curl_cffi を比較
"""
import os
import requests
from curl_cffi.requests import Session
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = os.getenv('FMP_API_KEY')
if not FMP_API_KEY:
    # .envファイルがない場合、環境変数から取得を試みる
    FMP_API_KEY = os.environ.get('FMP_API_KEY')
base_url = "https://financialmodelingprep.com/api/v3"

print(f"API Key (first 10 chars): {FMP_API_KEY[:10]}..." if FMP_API_KEY else "API Key not found")
print()

# テスト1: 株価データ取得
print("="*80)
print("テスト1: 株価データ取得 (historical-price-full)")
print("="*80)
test_symbol = "AAPL"
url = f"{base_url}/historical-price-full/{test_symbol}"
params = {'timeseries': 300, 'apikey': FMP_API_KEY}

print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print()

    if response.status_code == 200:
        data = response.json()
        print(f"Response Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            if 'historical' in data:
                print(f"Historical entries: {len(data['historical'])}")
                if data['historical']:
                    print(f"First entry: {data['historical'][0]}")
        else:
            print(f"Response: {data}")
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Exception: {e}")

print()

# テスト2: 損益計算書取得
print("="*80)
print("テスト2: 損益計算書取得 (income-statement)")
print("="*80)
url = f"{base_url}/income-statement/{test_symbol}"
params = {'period': 'quarter', 'limit': 8, 'apikey': FMP_API_KEY}

print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"Status Code: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print(f"Response Type: {type(data)}")
        print(f"Number of entries: {len(data) if isinstance(data, list) else 'N/A'}")
        if isinstance(data, list) and data:
            print(f"First entry keys: {list(data[0].keys())}")
            print(f"EPS in first entry: {data[0].get('eps', 'N/A')}")
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Exception: {e}")

print()

# テスト3: 企業プロファイル取得
print("="*80)
print("テスト3: 企業プロファイル取得 (profile)")
print("="*80)
url = f"{base_url}/profile/{test_symbol}"
params = {'apikey': FMP_API_KEY}

print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=30)
    print(f"Status Code: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print(f"Response Type: {type(data)}")
        if isinstance(data, list) and data:
            print(f"Profile keys: {list(data[0].keys())}")
            print(f"Company Name: {data[0].get('companyName', 'N/A')}")
            print(f"Sector: {data[0].get('sector', 'N/A')}")
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Exception: {e}")

print()
print()

# ==================== curl_cffi を使用したテスト ====================
print("="*80)
print("curl_cffi を使用したテスト")
print("="*80)
print()

# テスト4: curl_cffi で株価データ取得
print("="*80)
print("テスト4: curl_cffi で株価データ取得")
print("="*80)
url = f"{base_url}/historical-price-full/{test_symbol}"
params = {'timeseries': 300, 'apikey': FMP_API_KEY}

print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    session = Session(impersonate="chrome110")
    response = session.get(url, params=params)
    print(f"Status Code: {response.status_code}")
    print()

    if response.status_code == 200:
        data = response.json()
        print(f"Response Type: {type(data)}")
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            if 'historical' in data:
                print(f"Historical entries: {len(data['historical'])}")
                if data['historical']:
                    print(f"First entry: {data['historical'][0]}")
                    print("✓ SUCCESS with curl_cffi!")
        else:
            print(f"Response: {data}")
    else:
        print(f"Error Response: {response.text}")
except Exception as e:
    print(f"Exception: {e}")
