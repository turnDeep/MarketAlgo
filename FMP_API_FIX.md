# FMP API 修正内容

## 問題の原因

Financial Modeling Prep (FMP) API は Bot検出を行っており、標準の `requests` ライブラリからのリクエストを403エラー（Access Denied）でブロックしていました。

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

### 2. APIキーの設定
`.env` ファイルに有効なFMP APIキーを設定してください:
```bash
FMP_API_KEY=your_actual_api_key_here
```

**重要:** `demo` APIキーは動作しません。無料のAPIキーは以下から取得できます:
https://site.financialmodelingprep.com/developer/docs/

### 3. スクリプトの実行
```bash
python run_ibd_screeners.py
```

## 検証方法

修正が正しく動作するか確認するには:
```bash
python test_fmp_api.py
```

このスクリプトは `requests` と `curl_cffi` の両方でAPIをテストし、違いを示します。

## 注意事項

- FMP API には無料プランでもレート制限があります（300リクエスト/分）
- Premium プランの場合は750リクエスト/分まで利用可能
- `.env` ファイルで `FMP_RATE_LIMIT` を適切に設定してください

## 参考

- curl_cffi: https://github.com/yifeikong/curl_cffi
- FMP API ドキュメント: https://site.financialmodelingprep.com/developer/docs/
