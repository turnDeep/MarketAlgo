# FMP API 修正内容

## 問題の原因

Financial Modeling Prep (FMP) API でデータ取得に失敗していた原因は2つありました：

### 1. Bot検出による403エラー
FMP API は Bot検出を行っており、標準の `requests` ライブラリからのリクエストを403エラー（Access Denied）でブロックしていました。

### 2. 無効なAPIキー
`demo` APIキーは現在使用できなくなっています。有効なAPIキーが必要です。

### 症状
- 全銘柄のデータ取得が失敗（成功: 0 銘柄）
- ティッカーリストの取得は成功（`get_tickers.py` では `curl_cffi` を使用しているため）
- 株価データ、損益計算書、企業プロファイルの取得が全て失敗

## 修正内容

### `ibd_data_collector.py`
標準の `requests` ライブラリから `curl_cffi` ライブラリに変更しました。

**変更点:**
1. インポートを変更: `import requests` → `from curl_cffi.requests import Session`
2. `__init__` メソッドで curl_cffi セッションを初期化:
   ```python
   self.session = Session(impersonate="chrome110")
   ```
3. `fetch_with_rate_limit` メソッドで curl_cffi セッションを使用:
   ```python
   response = self.session.get(url, params=params, timeout=30)
   ```
4. `close` メソッドでセッションをクローズ

### なぜ `curl_cffi` が有効か
- `curl_cffi` は実際のブラウザ（Chrome 110）のTLS/HTTPフィンガープリントを模倣
- Bot検出システムを回避できる
- FMP API のような厳格なBot検出を行うAPIでも正常に動作

## 使用方法

### 1. 必要なパッケージのインストール
```bash
pip install curl_cffi
```

または、requirements.txtから:
```bash
pip install -r requirements.txt
```

### 2. APIキーの設定（重要！）

**現在、APIキーが `demo` のままだと動作しません。**

#### 方法A: 対話式スクリプトを使用
```bash
./setup_api_key.sh
```
指示に従ってAPIキーを入力してください。

#### 方法B: 手動で設定
`.env` ファイルを編集:
```bash
nano .env
```

以下のように変更:
```
FMP_API_KEY=your_actual_api_key_here
```

#### APIキーの取得方法
1. https://site.financialmodelingprep.com/developer/docs/ にアクセス
2. 無料アカウントを作成（メールアドレスのみで登録可能）
3. APIキーが自動的に生成されます
4. そのAPIキーを `.env` ファイルに設定

**無料プランでも十分使用可能です（300リクエスト/分）**

### 3. スクリプトの実行
```bash
python run_ibd_screeners.py
```

## 検証方法

### ステップ1: 基本的なAPIテスト
```bash
python test_fmp_api.py
```
このスクリプトは `requests` と `curl_cffi` の両方でAPIをテストし、違いを示します。

### ステップ2: 実際のデータ取得テスト
```bash
python test_single_ticker.py
```
AAPL（Apple Inc.）のデータを取得してAPIキーと修正が正しく機能しているか確認します。

**期待される出力:**
```
API Key: your_api_key_here
API Key length: 32

================================================================================
テスト1: 株価データ取得 (AAPL)
================================================================================
✓ 成功: 300 日分のデータを取得
  最新日付: 2025-11-12
  最新価格: $XXX.XX
...
```

**エラーの場合:**
- `Status: 403` → APIキーが無効または `demo` のまま
- `Status: 429` → レート制限に達しています（しばらく待機）
- その他のエラー → ネットワークまたはAPI側の問題

## 注意事項

- FMP API には無料プランでもレート制限があります（300リクエスト/分）
- Premium プランの場合は750リクエスト/分まで利用可能
- `.env` ファイルで `FMP_RATE_LIMIT` を適切に設定してください

## 参考

- curl_cffi: https://github.com/yifeikong/curl_cffi
- FMP API ドキュメント: https://site.financialmodelingprep.com/developer/docs/
