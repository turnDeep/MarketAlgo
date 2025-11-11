# Market Dashboard Files

このディレクトリには、`run_dashboard.py`に関連するすべてのファイルが含まれています。

## ファイル構成

### メインファイル
- **run_dashboard.py** - ダッシュボードの実行とJSON/HTML生成を行うメインランナー
- **market_dashboard.py** - マーケットダッシュボードジェネレーター（市場データ分析）
- **dashboard_visualizer.py** - HTML形式のダッシュボード生成

### データ取得・処理
- **fmp_data_fetcher.py** - FinancialModelingPrep APIからのデータ取得
- **data_fetcher.py** - 株価データ取得の汎用モジュール
- **oratnek_data_manager.py** - SQLiteベースのデータマネージャー

### 分析・計算モジュール
- **indicators.py** - テクニカル指標の計算（移動平均、RSI、MACDなど）
- **rs_calculator.py** - RS Rating（相対的強さ）の計算
- **stage_detector.py** - ステージ検出（Minervini手法）
- **oratnek_screeners.py** - IBD手法に基づく6つのスクリーニングリスト

### データファイル
- **stock.csv** - スクリーニング対象銘柄のリスト

## 依存関係

```
run_dashboard.py
├── market_dashboard.py
│   ├── fmp_data_fetcher.py
│   ├── data_fetcher.py
│   ├── indicators.py
│   ├── rs_calculator.py
│   ├── stage_detector.py
│   └── oratnek_screeners.py
│       ├── oratnek_data_manager.py
│       │   └── fmp_data_fetcher.py
│       ├── indicators.py
│       └── rs_calculator.py
└── dashboard_visualizer.py
```

## 使用方法

1. 環境変数の設定（オプション）
   `.env`ファイルにFinancialModelingPrep APIキーを設定してください。

2. ダッシュボードの実行
   ```bash
   python run_dashboard.py
   ```

3. 出力ファイル
   - `market_dashboard_data.json` - JSON形式のダッシュボードデータ
   - `market_dashboard.html` - HTML形式のダッシュボード
   - `market_exposure_history.csv` - Market Exposureの履歴データ

## 機能

### Market Dashboard (market_dashboard.py)
- Market Exposure（12要因評価）
- Market Performance Overview
- VIX Analysis
- Broad Market Overview
- Sector Analysis
- Power Law Indicators
- RS Rating Lists

### Oratnek Screeners (oratnek_screeners.py)
IBD手法に基づく6つのスクリーニング:
1. **Momentum 97** - 1M/3M/6Mすべてで上位3%
2. **Explosive EPS Growth** - 今四半期EPS予想が100%以上成長
3. **Up on Volume** - 出来高を伴って上昇している機関投資家注目銘柄
4. **Top 2% RS Rating** - RS Rating上位2%かつトレンドが完璧
5. **4% Bullish Yesterday** - 昨日4%以上上昇
6. **Healthy Chart Watch List** - 健全なチャート形状を持つ高品質銘柄

## 必要なパッケージ

```bash
pip install pandas numpy python-dotenv
```

## 注意事項

- FinancialModelingPrep APIキーが必要です
- 初回実行時は大量のデータをダウンロードするため時間がかかります
- データはSQLiteにキャッシュされます（`data/oratnek/oratnek_cache.db`）
