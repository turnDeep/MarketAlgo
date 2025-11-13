"""
実際に使用されているAPIキーを確認
"""
import os
from dotenv import load_dotenv

print("="*80)
print("APIキー設定の確認")
print("="*80)
print()

# 環境変数を直接確認（.envファイル読み込み前）
env_before = os.environ.get('FMP_API_KEY')
print(f"1. 環境変数（load_dotenv前）: {env_before[:10] + '...' if env_before and len(env_before) > 10 else env_before}")

# .envファイルを読み込み
load_dotenv()

# 環境変数を確認（.envファイル読み込み後）
env_after = os.getenv('FMP_API_KEY')
print(f"2. 環境変数（load_dotenv後）: {env_after[:10] + '...' if env_after and len(env_after) > 10 else env_after}")
print(f"   キーの長さ: {len(env_after) if env_after else 0} 文字")

# .envファイルの内容を確認
print()
print("3. .envファイルの内容:")
try:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('FMP_API_KEY'):
                key_value = line.split('=', 1)[1].strip()
                print(f"   {line.split('=', 1)[0]}={key_value[:10] + '...' if len(key_value) > 10 else key_value}")
                break
except FileNotFoundError:
    print("   .envファイルが見つかりません")

print()
print("="*80)
print("判定:")
print("="*80)

if env_after == 'demo':
    print("❌ 現在 'demo' キーが使用されています")
    print()
    print("解決方法:")
    print("1. .envファイルを編集して、プレミアムプランのAPIキーを設定してください")
    print("   nano .env")
    print()
    print("2. または、環境変数として設定してください:")
    print("   export FMP_API_KEY=your_premium_api_key")
elif env_after and len(env_after) > 10:
    print(f"✓ 有効なAPIキーが設定されています（{len(env_after)}文字）")
else:
    print("❌ APIキーが設定されていないか、短すぎます")
