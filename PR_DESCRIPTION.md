# Implement EPS Rating with Percentile Ranking (Option 2)

## 概要

このPRは、EPS Ratingの**正確な実装（オプション2）**を導入します。従来の実装では各銘柄ごとに個別にスコアを計算していましたが、新しい実装では**RS Ratingと同じパーセンタイルランキング方式**を採用しています。

## 主な変更点

### 1. SQLiteデータベースの導入

すべてのデータを集約管理するSQLiteデータベース（`ibd_data.db`）を導入：

- **テーブル構成**：
  - `tickers` - 銘柄マスター
  - `price_history` - 株価履歴
  - `income_statements_quarterly` - 四半期損益計算書
  - `income_statements_annual` - 年次損益計算書
  - `company_profiles` - 企業プロファイル
  - `calculated_rs` - 計算済みRS値
  - `calculated_eps` - 計算済みEPS要素
  - `calculated_ratings` - 最終レーティング

### 2. 新しいモジュール構成

#### `ibd_database.py`
- データベース管理クラス
- CRUD操作
- スキーマ定義

#### `ibd_data_collector.py`
- FMP APIからデータを収集してDBに保存
- 並列処理による高速化
- RS値とEPS要素の初期計算

#### `ibd_ratings_calculator.py`
- **パーセンタイルランキング方式によるレーティング計算**
- RS Rating計算
- EPS Rating計算（オプション2の核心）
- Composite Rating計算

#### `ibd_screeners_db.py`
- データベースから計算済みレーティングを取得
- 各IBDスクリーナーを実行
- Googleスプレッドシートに結果を出力

#### `run_ibd_screeners.py`
- 全ワークフローを統合した実行スクリプト

### 3. EPS Rating計算方式（オプション2）

#### 従来の方式（オプション1）
```
各銘柄ごとに:
  1. EPS成長率を計算
  2. 重み付けしてスコア化（0-100）
  3. そのまま使用
```
**問題点**: 銘柄間の相対的な位置が不明確

#### 新しい方式（オプション2 - RS Ratingと同様）
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

## 利点

1. **一貫性**: RS RatingとEPS Ratingで同じパーセンタイルランキング方式を使用
2. **相対評価**: 銘柄間の相対的な位置が明確
3. **データの再利用**: データベースに保存されたデータを再利用可能
4. **パフォーマンス**: 並列処理により高速化
5. **IBD準拠**: IBDの実装により近い

## 使用方法

### 全ステップ実行
```bash
python run_ibd_screeners.py
```

### 個別ステップ実行
```bash
# データ収集のみ
python run_ibd_screeners.py --collect-data

# レーティング計算のみ
python run_ibd_screeners.py --calculate-ratings

# スクリーナー実行のみ
python run_ibd_screeners.py --run-screeners

# テストモード（500銘柄に制限）
python run_ibd_screeners.py --test-mode
```

## ワークフロー

1. **データ収集**: 全銘柄の株価・EPS・プロファイルデータをFMP APIから取得してDBに保存
2. **レーティング計算**: RS Rating、EPS Rating（パーセンタイルランキング方式）、Composite Ratingを計算
3. **スクリーナー実行**: 各IBDスクリーナーを実行して結果をGoogleスプレッドシートに出力

## 追加ファイル

- `README_EPS_RATING_V2.md` - 詳細な実装ドキュメント
- `.env.example` - 新しい環境変数（IBD_DB_PATH）を追加
- `.gitignore` - データベースファイルを除外

## テスト

テストモードで動作確認可能（500銘柄に制限）

## 参考文献

- GitHub: skyte/relative-strength (RS Rating)
- William O'Neil + Co. の公式説明 (EPS Rating)
- IBD SmartSelect Corporate Ratings documentation (Composite Rating)

詳細は `README_EPS_RATING_V2.md` を参照してください。

## 変更統計

- 8ファイル変更
- 2,451行追加
- 新規ファイル: 5個
- 修正ファイル: 3個
