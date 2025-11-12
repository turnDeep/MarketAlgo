# IBD Stock Screeners

このプロジェクトは、Investor's Business Daily (IBD)スタイルの株式スクリーナーを実装しています。

## 実装されているスクリーナー

### 1. Momentum 97
最も勢いのある銘柄を検出します。

**条件:**
- 1M Rank (Pct) ≥ 97%
- 3M Rank (Pct) ≥ 97%
- 6M Rank (Pct) ≥ 97%

### 2. Explosive Estimated EPS Growth Stocks
爆発的なEPS成長が期待される銘柄を検出します。

**条件:**
- RS Rating ≥ 80
- EPS Est Cur Qtr % ≥ 100%
- 50-Day Avg Vol (1000s) ≥ 100
- Price vs 50-Day ≥ 0.0%

### 3. Up on Volume List
価格上昇と高ボリュームを伴う銘柄を検出します。

**条件:**
- Price % Chg ≥ 0.00%
- Vol% Chg vs 50-Day ≥ 20%
- Current Price ≥ $10
- 50-Day Avg Vol (1000s) ≥ 100
- Market Cap (mil) ≥ $250
- RS Rating ≥ 80
- EPS % Chg Last Qtr ≥ 20%
- A/D Rating ABC

### 4. Top 2% RS Rating List
最も強い相対強度を持つ銘柄を検出します。

**条件:**
- RS Rating ≥ 98
- 10Day > 21Day > 50Day
- 50-Day Avg Vol (1000s) ≥ 100
- Volume (1000s) ≥ 100
- Sector NOT: medical

### 5. 4% Bullish Yesterday
前日に大きく上昇した銘柄を検出します。

**条件:**
- Price ≥ $1
- Change > 4%
- Market cap > $250M
- Volume > 100K
- Rel Volume > 1
- Change from Open > 0%
- Avg Volume 90D > 100K

### 6. Healthy Chart Watch List
健全なチャートパターンを持つ銘柄を検出します。

**条件:**
- 10Day > 21Day > 50Day
- 50Day > 150Day > 200Day
- RS Line New High
- RS Rating ≥ 90
- A/D Rating AB
- Ind Group RS AB
- Comp Rating ≥ 80
- 50-Day Avg Vol (1000s) ≥ 100

## IBD指標の計算方法

### RS Rating (Relative Strength Rating)
IBD式のRS Ratingは、以下の加重平均で計算されます：

```
RS = 0.4 × ROC(63日) + 0.2 × ROC(126日) + 0.2 × ROC(189日) + 0.2 × ROC(252日)
```

- 最新の四半期（3ヶ月）に40%の重み
- 過去3四半期にそれぞれ20%の重み
- 全銘柄でパーセンタイルランキング（1-99）

### A/D Rating (Accumulation/Distribution Rating)
機関投資家の買い圧力/売り圧力を測定します。

- 13週間（65営業日）のデータを分析
- 価格上昇時のボリューム vs 価格下落時のボリュームの比率
- A（強い買い）からE（強い売り）まで評価

### Composite Rating
複数の指標を組み合わせた総合評価です：

- RS Rating: 40%
- EPS Growth: 30%
- A/D Rating: 20%
- 52週高値からの距離: 10%

## 使用方法

### 前提条件

1. Python 3.8以上
2. Financial Modeling Prep (FMP) APIキー（Premiumプラン推奨）
3. Google Cloud サービスアカウントの認証情報

### セットアップ

1. 必要なパッケージをインストール：
```bash
pip install gspread requests pandas numpy python-dotenv curl-cffi
```

2. `.env`ファイルを作成：
```bash
cp .env.example .env
```

3. `.env`ファイルを編集して、必要な情報を設定：
```
FMP_API_KEY=your_fmp_api_key_here
FMP_RATE_LIMIT=750
ORATNEK_MAX_WORKERS=10
CREDENTIALS_FILE=credentials.json
SPREADSHEET_NAME=Market Dashboard
```

4. Google Cloud サービスアカウントの認証情報JSON fileを配置：
```bash
# credentials.json をプロジェクトディレクトリに配置
```

### 実行

```bash
python ibd_screeners.py
```

## パフォーマンス最適化

### マルチスレッド処理
- デフォルトで10スレッドの並列処理を使用
- `ORATNEK_MAX_WORKERS`環境変数で調整可能
- スレッド数を増やすと処理が高速化しますが、API制限に注意

### レート制限
- FMP API Premium プランの制限: 750 calls/min
- スレッドセーフなレート制限機構を実装
- 自動的に待機して制限を超えないように調整

### 処理時間の目安
- 500銘柄: 約10-15分
- 5,000銘柄: 約1.5-2時間
- 全銘柄（7,000+）: 約2.5-3時間

## 出力フォーマット

Googleスプレッドシートに以下のフォーマットで出力されます：

```
[スクリーナー名]
TICKER1  TICKER2  TICKER3  ...  TICKER10
TICKER11 TICKER12 TICKER13 ...  TICKER20
...

[次のスクリーナー名]
TICKER1  TICKER2  TICKER3  ...  TICKER10
...
```

- 各スクリーナー名の下にティッカーを横10個ずつ配置
- スクリーナー間に空行を挿入
- ヘッダー行は青色でハイライト

## 注意事項

### API制限
- FMP API の Premium プランが必要です（750 calls/min）
- Starter プランでは処理が非常に遅くなります（300 calls/min）
- Professional プランでは処理が高速化します（1,500 calls/min）

### データの正確性
- IBDの正確な計算式は一部非公開のため、近似値を使用
- 特にComposite RatingとA/D Ratingは推定値
- RS Ratingは公開されている計算式に基づいて正確に計算

### メモリ使用量
- 全銘柄を処理する場合、約1-2GBのメモリを使用
- 大量の価格データをキャッシュするため

## トラブルシューティング

### API エラー
```
Error: API request failed
```
- API キーが正しいか確認
- API制限を超えていないか確認
- インターネット接続を確認

### Google Sheets エラー
```
Error: Spreadsheet not found
```
- `credentials.json` ファイルが正しい場所にあるか確認
- サービスアカウントに必要な権限があるか確認
- スプレッドシート名が正しいか確認

### メモリエラー
```
MemoryError
```
- `use_full_dataset=False` でテストモードを使用
- マシンのメモリを増やす
- バッチサイズを小さくする

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 免責事項

このツールは教育目的で作成されています。投資判断は自己責任で行ってください。
IBDの商標および手法は William O'Neil + Co. に帰属します。
