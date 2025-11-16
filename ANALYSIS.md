# RS STS% フィルター問題の分析

## 発見された問題

### 1. **データベースの問題（最も可能性が高い）**

現在の開発環境ではデータベースが空（0バイト）のため、`price_history`テーブルにデータがない可能性が高いです。

**症状**: 全ての銘柄で`get_rs_sts_percentile()`が`None`を返す

**原因**:
- `benchmark_prices is None` (SPYのデータがない)
- `ticker_prices is None` (個別銘柄のデータがない)
- データが25日分未満

**確認方法**:
```bash
# デバッグモードで実行
export IBD_DEBUG=true
python3 run_ibd_screeners.py
```

### 2. **日付の不一致の問題**

`calculate_relative_strength`メソッドでは、benchmarkとtargetの価格配列を単純に割り算していますが、両方のDataFrameの日付が一致しているかを確認していません。

**問題のコード** (`ibd_screeners.py:146-158`):
```python
benchmark = benchmark_prices['close'].tail(min_len).values
target = target_prices['close'].tail(min_len).values

# 日付の確認なし！
rs = target / benchmark
```

もし、benchmarkとtargetで異なる取引日がある場合、RSの計算が誤ります。

**例**:
- SPY: [2025-11-01, 2025-11-02, 2025-11-03, ...]（アメリカ市場）
- 個別銘柄: [2025-11-01, 2025-11-03, 2025-11-04, ...]（一部の日が欠損）

この場合、配列のインデックスが一致せず、誤ったRSが計算されます。

### 3. **フィルター条件が厳しすぎる**

RS STS% >= 80という条件は、最新のRS値が過去25日間で**上位20%**に入っている必要があります。

**具体的には**:
- 過去25日間のRS値のうち、最新のRS値が最も高い5日分の中に入っている必要がある
- つまり、最近の相対パフォーマンスが非常に良い銘柄のみが通過する

**市場が全体的に下落している場合**、ほとんどの銘柄がこの条件を満たさない可能性があります。

### 4. **RS値の定義の問題の可能性**

現在の実装では、RS値を単純に`ticker価格 / SPY価格`として計算していますが、IBDの「RS value」の定義が異なる可能性があります。

**IBDのRS Rating**は、以下の要素を組み合わせた複雑な計算です:
- 直近3ヶ月、6ヶ月、9ヶ月、12ヶ月のパフォーマンス
- 相対的なランキング（パーセンタイル）

現在の実装は、単純な価格比率なので、IBDの定義と異なる可能性があります。

## 推奨される対策

### 対策1: データベースの確認

```bash
# データベースのテーブルを確認
python3 check_db_schema.py

# price_historyテーブルのデータ数を確認
python3 -c "
import sqlite3
conn = sqlite3.connect('ibd_data.db')
cursor = conn.cursor()
cursor.execute('SELECT ticker, COUNT(*) FROM price_history GROUP BY ticker LIMIT 10')
print(cursor.fetchall())
"
```

### 対策2: 日付の不一致を修正

`calculate_relative_strength`メソッドを修正して、日付でマージするように変更:

```python
def calculate_relative_strength(self, benchmark_prices, target_prices, days=25):
    if benchmark_prices is None or target_prices is None:
        return None

    # 日付でマージして共通の日付のみを使用
    merged = pd.merge(
        benchmark_prices[['date', 'close']].rename(columns={'close': 'benchmark_close'}),
        target_prices[['date', 'close']].rename(columns={'close': 'target_close'}),
        on='date',
        how='inner'
    )

    if len(merged) < days:
        return None

    # 最新のdays日分を使用
    merged = merged.tail(days)

    # ゼロ除算を防ぐ
    if (merged['benchmark_close'] == 0).any():
        return None

    rs = merged['target_close'].values / merged['benchmark_close'].values
    return rs
```

### 対策3: フィルター条件の緩和（オプション）

もし条件が厳しすぎる場合、以下のオプションがあります:

1. **RS STS%の閾値を下げる**: 80 → 70 または 60
2. **データがない場合はスキップしない**: `rs_sts is None`の場合、フィルターを適用しない
3. **RS STS%の計算方法を変更**: より緩やかな計算方法を使用

### 対策4: RS値の定義を確認

IBDのRS STS%の正確な定義を確認して、実装が正しいかを検証する必要があります。

## デバッグ方法

1. **デバッグモードで実行**:
```bash
export IBD_DEBUG=true
python3 run_ibd_screeners.py
```

2. **個別銘柄のRS STS%を計算**:
```python
from ibd_screeners import IBDScreeners

screener = IBDScreeners('credentials.json', 'Your Spreadsheet', 'ibd_data.db')
rs_sts = screener.get_rs_sts_percentile('AAPL', debug=True)
print(f"AAPL RS STS%: {rs_sts}")
```

3. **price_historyのデータを確認**:
```python
from ibd_database import IBDDatabase

db = IBDDatabase('ibd_data.db')
spy_prices = db.get_price_history('SPY', days=30)
aapl_prices = db.get_price_history('AAPL', days=30)

print(f"SPY: {len(spy_prices) if spy_prices is not None else 'None'} days")
print(f"AAPL: {len(aapl_prices) if aapl_prices is not None else 'None'} days")

if spy_prices is not None:
    print(f"SPY date range: {spy_prices['date'].min()} to {spy_prices['date'].max()}")
if aapl_prices is not None:
    print(f"AAPL date range: {aapl_prices['date'].min()} to {aapl_prices['date'].max()}")
```

## 次のステップ

1. **データベースの状態を確認**して、price_historyテーブルにデータがあるかを確認
2. **デバッグモードで実行**して、どこで問題が発生しているかを特定
3. **問題に応じて適切な対策を実施**:
   - データがない → データを収集
   - 日付の不一致 → `calculate_relative_strength`を修正
   - 条件が厳しすぎる → 閾値を調整
   - RS値の定義が違う → 計算方法を修正
