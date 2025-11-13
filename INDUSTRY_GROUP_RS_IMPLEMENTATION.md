# Industry Group RS実装ドキュメント

## 概要

Industry Group RS（業界グループ相対強度）をFinancialModelingPrep（FMP）APIを使用して実装しました。

## Industry Group RSとは

Industry Group RSは、個別銘柄のパフォーマンスとその業界グループ（セクター）のパフォーマンスを比較した相対強度指標です。IBDの評価システムにおいて、Composite Ratingの計算に10%のウェイトで使用されます。

## 実装内容

### 1. データベーススキーマの拡張

#### 新しいテーブル

**sector_performance**
- セクターの履歴パフォーマンスデータを保存
- カラム: id, sector, date, change_percentage

**calculated_industry_group_rs**
- 各銘柄のIndustry Group RS計算結果を保存
- カラム: ticker, sector, industry, stock_rs_value, sector_rs_value, industry_group_rs_value, calculated_at

**calculated_ratings（更新）**
- industry_group_rsカラムを追加

### 2. データ収集機能（ibd_data_collector.py）

#### 新しいメソッド

**get_historical_sector_performance()**
- FMP API: `/api/v3/historical-sectors-performance`
- 履歴セクターパフォーマンスデータを取得

**get_current_sector_performance()**
- FMP API: `/api/v3/sectors-performance`
- 現在のセクターパフォーマンスを取得

**collect_sector_performance_data()**
- セクターパフォーマンスデータを収集してDBに保存
- 重複を除去して効率的に保存

### 3. Industry Group RS計算（ibd_ratings_calculator.py）

#### 計算ロジック

```python
Industry Group RS = 銘柄RS値 - セクターRS値
```

- **正の値**: 銘柄がセクターをアウトパフォーム
- **負の値**: 銘柄がセクターをアンダーパフォーム

#### 新しいメソッド

**calculate_sector_rs_value(sector_perf_df)**
- セクターのRS値を株価RS値と同じ方式で計算
- IBD式の加重平均を使用（最新四半期に40%の重み）
- 計算式: `0.4 * ROC_63d + 0.2 * ROC_126d + 0.2 * ROC_189d + 0.2 * ROC_252d`

**calculate_industry_group_rs(ticker)**
- 単一銘柄のIndustry Group RSを計算
- 1. 銘柄のセクター情報を取得
- 2. 銘柄のRS値を取得
- 3. セクターパフォーマンスデータを取得
- 4. セクターのRS値を計算
- 5. Industry Group RS = 銘柄RS - セクターRS

**calculate_all_industry_group_rs()**
- 全銘柄のIndustry Group RSを計算
- パーセンタイルランキング（0-100）に変換
- データベースに保存

### 4. Composite Ratingへの統合

**calculate_comp_rating()** メソッドを更新して、Industry Group RSを10%のウェイトで含めるようにしました。

#### Composite Ratingのウェイト配分
- RS Rating: 30%
- EPS Rating: 30%
- A/D Rating: 15%
- SMR Rating: 10%
- **Industry Group RS: 10%** ← 新規追加
- 52週高値からの距離: 5%

### 5. データベース操作メソッド（ibd_database.py）

**セクターパフォーマンス関連**
- `insert_sector_performance()`: 単一レコード挿入
- `insert_sector_performance_bulk()`: 一括挿入
- `get_sector_performance_history()`: 履歴データ取得
- `get_all_sectors()`: 全セクター取得

**Industry Group RS関連**
- `insert_industry_group_rs()`: Industry Group RSデータ挿入
- `get_all_industry_group_rs()`: 全銘柄のIndustry Group RS値取得
- `get_industry_group_rs()`: 特定銘柄のIndustry Group RS取得

## 使用方法

### 1. セクターパフォーマンスデータの収集

```python
from ibd_data_collector import IBDDataCollector

collector = IBDDataCollector(fmp_api_key)
collector.collect_sector_performance_data(limit=300)
collector.close()
```

### 2. Industry Group RSの計算

```python
from ibd_ratings_calculator import IBDRatingsCalculator

calculator = IBDRatingsCalculator()

# 単一銘柄
ig_rs_data = calculator.calculate_industry_group_rs('AAPL')
print(f"Industry Group RS: {ig_rs_data['industry_group_rs_value']}")

# 全銘柄
ig_rs_ratings = calculator.calculate_all_industry_group_rs()

calculator.close()
```

### 3. 全レーティングの計算（Industry Group RSを含む）

```python
calculator = IBDRatingsCalculator()
calculator.calculate_all_ratings()  # Industry Group RSも自動的に計算されます
calculator.close()
```

### 4. テストスクリプトの実行

```bash
python test_industry_group_rs.py
```

## FMP API エンドポイント

### Historical Sector Performance
- **エンドポイント**: `https://financialmodelingprep.com/api/v3/historical-sectors-performance`
- **パラメータ**:
  - `limit`: 取得する履歴データの件数
  - `apikey`: APIキー

### Current Sector Performance
- **エンドポイント**: `https://financialmodelingprep.com/api/v3/sectors-performance`
- **パラメータ**:
  - `apikey`: APIキー

## データフロー

```
1. FMP API
   ↓
2. セクターパフォーマンスデータ収集
   ↓
3. sector_performance テーブルに保存
   ↓
4. セクターRS値計算
   ↓
5. Industry Group RS計算
   ↓
6. calculated_industry_group_rs テーブルに保存
   ↓
7. パーセンタイルランキングに変換
   ↓
8. Composite Ratingに統合
   ↓
9. calculated_ratings テーブルに保存
```

## 注意事項

1. **データ要件**
   - 最低252営業日の株価データが必要
   - 最低252日分のセクターパフォーマンスデータが必要
   - 企業プロファイルにセクター情報が必要

2. **API制限**
   - FMP APIのレート制限に注意（1分間に750リクエスト）
   - セクターパフォーマンスデータは定期的に更新が必要

3. **計算時間**
   - 全銘柄の計算には時間がかかります
   - データベースにインデックスが作成されているため、クエリは高速です

## ファイル変更一覧

- `ibd_database.py`: データベーススキーマとメソッドを追加
- `ibd_data_collector.py`: セクターパフォーマンスデータ収集メソッドを追加
- `ibd_ratings_calculator.py`: Industry Group RS計算メソッドを追加
- `test_industry_group_rs.py`: テストスクリプト（新規作成）
- `INDUSTRY_GROUP_RS_IMPLEMENTATION.md`: このドキュメント（新規作成）

## 参考文献

- [FMP Historical Sector Performance API](https://site.financialmodelingprep.com/developer/docs/stable/historical-sector-performance)
- [FMP Sector Performance API](https://site.financialmodelingprep.com/developer/docs/stock-market-sector-performance-free-api)
- IBD (Investor's Business Daily) Rating System

## 今後の改善点

1. **業界（Industry）レベルのRS**
   - 現在はセクターレベルのみ
   - より細かい業界レベルでの相対強度も計算可能

2. **パフォーマンス最適化**
   - セクターRS値のキャッシング
   - 並列処理の導入

3. **データ更新**
   - セクターパフォーマンスデータの自動更新機能
   - 増分更新（差分のみ取得）

4. **可視化**
   - セクター別パフォーマンスチャート
   - Industry Group RS分布ヒストグラム
