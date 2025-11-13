# EPS Rating パーセンタイルランキング実装（オプション2）

このドキュメントは、IBD EPS Ratingの**正確な実装（オプション2）**について説明します。

## 概要

従来のEPS Rating実装では、各銘柄ごとに個別にスコア（0-100）を計算していました。
新しい実装では、**RS Ratingと同じ方式**を採用し、全銘柄のEPSデータを収集して、
各要素を個別にパーセンタイルランキングし、重み付けして最終ランキングを計算します。

## アーキテクチャ

### 1. データベース設計（SQLite）

すべてのデータをSQLiteデータベース（`ibd_data.db`）に集約します：

#### テーブル構成

1. **tickers** - 銘柄マスター
2. **price_history** - 株価履歴（日次OHLCV）
3. **income_statements_quarterly** - 四半期損益計算書
4. **income_statements_annual** - 年次損益計算書
5. **company_profiles** - 企業プロファイル
6. **calculated_rs** - 計算済みRS値
7. **calculated_eps** - 計算済みEPS要素（4つの独立した要素）
8. **calculated_ratings** - 最終レーティング（RS Rating, EPS Rating, Composite Rating）

### 2. モジュール構成

#### ibd_database.py
- データベース管理クラス
- CRUD操作
- スキーマ定義

#### ibd_data_collector.py
- FMP APIからデータを取得してDBに保存
- 並列処理でデータ収集を高速化
- RS値とEPS要素の初期計算

#### ibd_ratings_calculator.py
- **パーセンタイルランキング方式によるレーティング計算**
- RS Rating計算
- EPS Rating計算（オプション2の核心）
- Composite Rating計算

#### ibd_screeners_db.py
- データベースから計算済みレーティングを取得
- 各IBDスクリーナーを実行
- Googleスプレッドシートに結果を出力

#### run_ibd_screeners.py
- 全ワークフローを統合した実行スクリプト

## EPS Rating計算方式（オプション2）

### 従来の方式（オプション1）

```
各銘柄ごとに:
  1. EPS成長率を計算
  2. 重み付けしてスコア化（0-100）
  3. そのまま使用
```

問題点：銘柄間の相対的な位置が不明確

### 新しい方式（オプション2 - RS Ratingと同様）

```
全銘柄のデータを収集:
  1. 各銘柄のEPS要素を計算
     - 最新四半期EPS成長率（前年同期比）
     - 前四半期EPS成長率（前年同期比）
     - 年間EPS成長率（3年CAGR）
     - 収益安定性スコア（変動係数）

  2. 各要素を個別にパーセンタイルランキング（0-100）
     - 最新四半期成長率: 全銘柄をランク付け
     - 前四半期成長率: 全銘柄をランク付け
     - 年間成長率: 全銘柄をランク付け
     - 安定性スコア: 全銘柄をランク付け

  3. 重み付けして最終EPSランキングを計算
     - 最新四半期: 50%
     - 前四半期: 20%
     - 年間成長率: 20%
     - 安定性: 10%
```

利点：
- RS Ratingと同じ方式で一貫性がある
- 銘柄間の相対的な位置が明確
- IBDの実装に近い

## ワークフロー

### ステップ1: データ収集

```bash
python run_ibd_screeners.py --collect-data
```

全銘柄（NASDAQ + NYSE）のデータを取得してDBに保存：
- 株価データ（300日分）
- 四半期損益計算書（8四半期）
- 年次損益計算書（5年分）
- 企業プロファイル

### ステップ2: レーティング計算

```bash
python run_ibd_screeners.py --calculate-ratings
```

全銘柄のレーティングを計算：

1. **RS値の計算**
   ```
   RS = 0.4 * ROC(63d) + 0.2 * ROC(126d) + 0.2 * ROC(189d) + 0.2 * ROC(252d)
   ```

2. **EPS要素の計算**
   - 最新四半期EPS成長率
   - 前四半期EPS成長率
   - 年間EPS成長率（CAGR）
   - 収益安定性スコア

3. **パーセンタイルランキング変換**
   - RS値 → RS Rating (0-100)
   - 各EPS要素 → 個別にパーセンタイル (0-100)

4. **EPS Ratingの計算**
   ```
   EPS Rating =
     0.5 * Percentile(最新四半期成長率) +
     0.2 * Percentile(前四半期成長率) +
     0.2 * Percentile(年間成長率) +
     0.1 * Percentile(安定性スコア)
   ```

5. **Composite Ratingの計算**
   ```
   Composite Rating =
     0.30 * RS Rating +
     0.30 * EPS Rating +
     0.15 * A/D Rating +
     0.05 * 52週高値スコア +
     0.10 * Industry Group RS (未実装) +
     0.10 * SMR Rating (未実装)
   ```

### ステップ3: スクリーナー実行

```bash
python run_ibd_screeners.py --run-screeners
```

6つのIBDスクリーナーを実行して結果をGoogleスプレッドシートに出力：
1. Momentum 97
2. Explosive Estimated EPS Growth Stocks
3. Up on Volume List
4. Top 2% RS Rating List
5. 4% Bullish Yesterday
6. Healthy Chart Watch List

### 全ステップ実行

```bash
python run_ibd_screeners.py
```

全ステップを順次実行します。

## テストモード

テストモード（500銘柄に制限）で実行：

```bash
python run_ibd_screeners.py --test-mode
```

## データベース統計

データベースの統計情報を確認：

```python
from ibd_database import IBDDatabase

db = IBDDatabase('ibd_data.db')
db.get_database_stats()
db.close()
```

## EPS Rating計算例

### 銘柄A

**EPS要素（生の値）：**
- 最新四半期成長率: 150%
- 前四半期成長率: 80%
- 年間成長率: 30%
- 安定性スコア: 85

**全銘柄中のパーセンタイル：**
- 最新四半期: 95位（全体の95%）
- 前四半期: 88位
- 年間成長率: 75位
- 安定性: 82位

**最終EPS Rating：**
```
EPS Rating = 0.5 * 95 + 0.2 * 88 + 0.2 * 75 + 0.1 * 82
           = 47.5 + 17.6 + 15.0 + 8.2
           = 88.3
```

## 利点

### 1. 一貫性
RS RatingとEPS Ratingで同じパーセンタイルランキング方式を使用

### 2. 相対評価
銘柄間の相対的な位置が明確

### 3. データの再利用
データベースに保存されたデータを再利用可能

### 4. パフォーマンス
並列処理により高速化

### 5. IBD準拠
IBDの実装により近い

## 制限事項

1. **FMP API制限**
   - レート制限: 750 calls/minute
   - データの正確性: APIプロバイダに依存

2. **未実装の要素**
   - Industry Group RS Rating
   - SMR Rating

3. **近似実装**
   - IBDの正確な計算式は企業秘密
   - コミュニティ研究に基づく推定実装

## 環境変数

`.env`ファイルに以下を設定：

```bash
# FMP API
FMP_API_KEY=your_api_key_here

# Google Sheets
CREDENTIALS_FILE=credentials.json
SPREADSHEET_NAME=Market Dashboard

# Settings
ORATNEK_MAX_WORKERS=10
IBD_DB_PATH=ibd_data.db
```

## 依存関係

```bash
pip install -r requirements.txt
```

主要なライブラリ：
- pandas
- numpy
- requests
- gspread
- python-dotenv
- sqlite3（標準ライブラリ）

## 参考文献

1. **RS Rating**
   - GitHub: skyte/relative-strength
   - Medium: "Calculating the IBD RS Rating with Python" by Shashank Vemuri

2. **EPS Rating**
   - William O'Neil + Co. の公式説明
   - "How to Make Money in Stocks" (William O'Neil著)

3. **Composite Rating**
   - IBD SmartSelect Corporate Ratings documentation
   - Community reverse-engineering efforts

## ライセンス

教育・研究目的のみ。投資判断の唯一の根拠とすべきではありません。
