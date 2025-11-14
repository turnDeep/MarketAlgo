# IBDと有志コミュニティにおける赤字企業の扱い - 調査レポート

## 調査日: 2025-11-14

---

## エグゼクティブサマリー

IBD（Investor's Business Daily）とCAN SLIMコミュニティにおける**年間成長率**と**安定性スコア**の計算方法、特に**赤字企業の扱い**について調査しました。

### 主要な発見:

1. **IBD EPS Ratingは赤字企業にも低いレーティングを付与する**（完全除外ではない）
2. **CAN SLIMスクリーニングでは赤字企業を除外する**（正のEPSが必須条件）
3. **年間成長率の計算には正のEPSが必要**（過去の赤字は許容されるが重み付けで不利）
4. **安定性スコアは変動係数（CV）を使用**（標準的な統計手法）

---

## 1. IBD EPS Ratingにおける赤字企業の扱い

### 1.1. 基本方針

**IBDは赤字企業を完全に除外しない** - むしろ低いレーティングを付与する

> "Companies that never achieved positive earnings since their IPO result in a low Earnings Per Share Rating, though the rating can begin to increase once analysts forecast positive future estimates."
>
> (出典: Deepvue, IBD EPS Rating)

### 1.2. 計算方法

IBD EPS Ratingは**4つの要素**を個別にランキングし、重み付けして統合：

1. **最新四半期のEPS成長率**（前年同期比）
2. **前四半期のEPS成長率**（前年同期比）
3. **年間EPS成長率**（5年CAGR、利用不可の場合は3年CAGR）
4. **収益安定性ファクター**（四半期EPSのボラティリティ）

**重要**: 各要素を個別にランキングしてから重み付けするため、一部のデータが欠損していても計算可能

### 1.3. 赤字企業のレーティング例

**DoorDash (DASH) - 2024年の例**:
- IPO以降、正のEPSを達成したことがない
- 結果: **低いEPS Rating**
- ただし、アナリストが将来の正のEPS予測を出すと、レーティングが上昇し始める

**Intuitive Surgical - 歴史的事例**:
- 2001-2003年に赤字
- 2004年以降、堅調な利益
- 結果: EPS Rating **78**（過去の赤字が重しとなったが、除外はされていない）

### 1.4. 推奨基準

- **通常の成長株**: EPS Rating 90以上
- **ターンアラウンド株・景気循環株**: EPS Rating 70以上でも許容
- **新規IPO**: データ不足で低レーティングでも除外しない

---

## 2. CAN SLIMスクリーニングにおける赤字企業の扱い

### 2.1. 厳格な除外基準

**CAN SLIMスクリーニングでは正のEPSが必須**

> "Positive earnings for the current quarter are required to help make the results of the growth rate calculation more meaningful."
>
> (出典: AAII, O'Neil's CAN SLIM Screen)

### 2.2. 除外理由: 数学的な問題

**負のEPSでは成長率計算が無意味になる**:

```python
# 例: EPSが改善しているが両方負の場合
EPSt-1 = -10
EPSt = -15  # 実際には損失が拡大している

# 成長率計算
growth = (EPSt - EPSt-1) / EPSt-1
       = (-15 - (-10)) / (-10)
       = -5 / -10
       = +0.5  # プラス50%成長！？（誤った結果）
```

この計算では「損失が拡大している」のに「成長率がプラス」になってしまう。

### 2.3. CAN SLIMの具体的な基準

**AAII O'Neil's CAN SLIM Revised 3rd Edition Screen**の基準:

#### 四半期EPS成長率
- **最低**: 18-20%成長（前年同期比）
- **必須条件**: 当四半期のEPSが正であること
- **対象**: 継続事業からのEPSのみ（一時的な項目は除外）

#### 年間EPS成長率
- **必須条件**: 過去3年間、毎年EPSが増加していること
  - 旧版（第2版）では過去5年間だったが、第3版で緩和
- **推奨成長率**: 過去3年間で25%以上の年間成長率

### 2.4. CAN SLIMの哲学

> "Companies with negative earnings don't experience real growth until earnings become positive."
>
> (出典: AAII)

**結論**: CAN SLIMでは、連続して赤字の企業は成長率を計算する意味がないと判断

---

## 3. 年間成長率の計算方法

### 3.1. IBDの公式アプローチ

**5年CAGR（利用不可の場合は3年CAGR）**:

```python
# 3年CAGRの計算例
EPS_latest = 5.00  # 最新年度のEPS
EPS_3years_ago = 3.00  # 3年前のEPS
years = 3

CAGR = (EPS_latest / EPS_3years_ago) ** (1/years) - 1
     = (5.00 / 3.00) ** (1/3) - 1
     = 1.186 ** 0.333 - 1
     = 0.186 = 18.6%
```

### 3.2. 過去の赤字の扱い

**IBD方式**:
- 過去の赤字は**ペナルティとして重み付けに影響**
- ただし、**最近のパフォーマンスに重点**を置く
- 過去3年間の年間EPSと直近2四半期に最も重きを置く

**CAN SLIM方式**:
- 過去3年間、**毎年EPSが増加**していることが必須
- つまり、過去3年以内に赤字があれば除外

### 3.3. コミュニティ実装

**GitHub等のオープンソース実装**では:
- IBD EPS Ratingの完全な実装は**ほとんど公開されていない**（プロプライエタリ）
- **IBD RS Rating**（相対強度）の実装は多数存在
- ほとんどの実装が**正のEPSを前提**としている

---

## 4. 安定性スコアの計算方法

### 4.1. IBDの安定性ファクター

**IBD公式**:
- "Lower numbers represent more stable company earnings history"
- "Higher numbers represent a tendency for more volatile or unpredictable earnings history"
- "Stocks with consistent earnings reports and growth tend to have strong earnings stability"

**重要**: 安定性ファクターは**他の要素より重み付けが低い**

### 4.2. 標準的な測定方法: 変動係数（Coefficient of Variation, CV）

**変動係数の計算**:

```python
import numpy as np

# 過去8四半期のEPSデータ
eps_data = [2.1, 2.3, 2.0, 2.4, 2.2, 2.5, 2.3, 2.6]

# 平均と標準偏差
mean_eps = np.mean(eps_data)  # 2.3
std_eps = np.std(eps_data)    # 0.19

# 変動係数
cv = std_eps / mean_eps
   = 0.19 / 2.3
   = 0.083 (8.3%)
```

**解釈**:
- **CV < 0.15**: 安定した収益（優良）
- **CV 0.15-0.30**: 中程度のボラティリティ
- **CV > 0.30**: 高いボラティリティ（不安定）

### 4.3. 学術的な裏付け

変動係数は**広く使用されている標準的な手法**:

- Minton and Schrand (1999)
- Minton, Schrand, and Walther (2002)
- Diether et al. (2002)

> "The coefficient of variation is obtained by dividing the standard deviation in earnings over five years by the average earnings over the period."
>
> (出典: NYU Stern, Aswath Damodaran)

### 4.4. 負のEPSを含む場合の問題

**CVは正の値を前提としている**:

```python
# 負のEPSを含む場合
eps_data = [2.1, -0.5, 2.0, 2.4, 2.2, -1.0, 2.3, 2.6]

mean_eps = np.mean(eps_data)  # 1.51
std_eps = np.std(eps_data)    # 1.42

cv = std_eps / mean_eps
   = 1.42 / 1.51
   = 0.94 (94%)  # 非常に高い
```

**平均が負になる場合、CVは無意味**:

```python
# 平均が負の場合
eps_data = [-0.5, -1.0, -0.8, 0.2, -0.3, -0.6, -0.9, -0.4]

mean_eps = np.mean(eps_data)  # -0.54
cv = std_eps / mean_eps       # 負の値で割る → 無意味
```

---

## 5. 有志コミュニティの実装例

### 5.1. AAII（American Association of Individual Investors）

**O'Neil's CAN SLIM Screens**:
- 複数のバージョンを提供（Revised 3rd Edition, No Float, etc.）
- **すべて正のEPSが前提**
- 年間成長率: 過去3年間毎年増加（第3版）、過去5年間毎年増加（旧版）
- 推奨年間成長率: 25%以上

### 5.2. Portfolio123

**CAN SLIM実装**:
- 4つの要素を個別にランキング
- 正のEPSが必須
- 詳細な計算式は会員限定

### 5.3. GitHub実装

**github.com/skyte/relative-strength**:
- IBD RS Rating（相対強度）の実装
- EPS Ratingの実装は**含まれていない**

**github.com/nickklosterman/IBD**:
- IBD関連のスクリーニングツール
- EPS Ratingの完全な実装は**非公開**

### 5.4. コミュニティのコンセンサス

**Reddit、Elite Trader、AmiBrokerフォーラム**での議論:
- IBDの正確な計算式は**プロプライエタリ**
- ほとんどの実装者が**正のEPSを前提**
- 赤字企業は「低いレーティング」または「除外」のいずれか

---

## 6. 現在の実装との比較

### 6.1. 当プロジェクトの現在の実装

**年間成長率** (`ibd_data_collector.py:377-389`):

```python
# 現在の実装（非常に厳格）
if income_statements_annual and len(income_statements_annual) >= 3:
    eps_values = [stmt.get('eps', 0) for stmt in income_statements_annual[:3]]
    # 全期間でEPSが正の値である必要がある
    if eps_values[0] and eps_values[0] > 0 and eps_values[-1] and eps_values[-1] > 0:
        years = len(eps_values) - 1
        cagr = (pow(eps_values[0] / eps_values[-1], 1/years) - 1) * 100
        annual_growth_rate = cagr
```

**安定性スコア** (`ibd_data_collector.py:392-410`):

```python
# 現在の実装（厳格）
if len(income_statements_quarterly) >= 8:
    eps_last_8q = [stmt.get('eps', 0) for stmt in income_statements_quarterly[:8]]
    # 正のEPSのみフィルタ
    eps_last_8q = [e for e in eps_last_8q if e is not None and e > 0]

    # 少なくとも6期のEPSが正の値である必要がある
    if len(eps_last_8q) >= 6:
        eps_mean = np.mean(eps_last_8q)
        eps_std = np.std(eps_last_8q)

        if eps_mean > 0:
            coefficient_of_variation = eps_std / eps_mean
            # スコアに変換（0-100、低いCVほど高スコア）
            stability_score = max(0, 100 - (coefficient_of_variation * 100))
```

### 6.2. IBD/CAN SLIMとの整合性

| 項目 | 当実装 | IBD | CAN SLIM | 評価 |
|------|--------|-----|----------|------|
| **年間成長率の計算** | 3年CAGR | 5年（または3年）CAGR | 過去3年毎年増加 | ✅ 整合性あり |
| **正のEPS要件** | 全期間で正のEPS | 過去の赤字は許容（ペナルティ） | 必須 | ⚠️ CAN SLIMに近い（厳格） |
| **安定性スコアの計算** | 変動係数（CV） | 不明（プロプライエタリ） | - | ✅ 学術的に妥当 |
| **負のEPSフィルタ** | 8期中6期以上正のEPS | 低レーティング付与 | 完全除外 | ⚠️ 中間的（やや厳格） |

---

## 7. 推奨される改善策

### 7.1. 年間成長率の計算改善

#### オプション1: IBD方式（過去の赤字を許容）

```python
def calculate_annual_growth_rate_ibd_style(income_statements_annual):
    """
    IBD方式: 過去に赤字があってもCAGRを計算し、ペナルティを適用
    """
    if not income_statements_annual or len(income_statements_annual) < 3:
        return None

    eps_values = [stmt.get('eps', 0) for stmt in income_statements_annual[:3]]

    # 最新と最古のEPSが正であれば計算（中間に赤字があってもOK）
    if eps_values[0] > 0 and eps_values[-1] > 0:
        years = len(eps_values) - 1
        cagr = (pow(eps_values[0] / eps_values[-1], 1/years) - 1) * 100

        # 中間年に赤字があればペナルティ
        negative_years = sum(1 for eps in eps_values[1:-1] if eps <= 0)
        penalty_factor = 1 - (negative_years * 0.2)  # 赤字1年ごとに20%減点

        return cagr * penalty_factor

    return None
```

#### オプション2: CAN SLIM方式（厳格に除外）

```python
def calculate_annual_growth_rate_canslim_style(income_statements_annual):
    """
    CAN SLIM方式: 過去3年間、毎年EPSが増加していることが必須
    """
    if not income_statements_annual or len(income_statements_annual) < 3:
        return None

    eps_values = [stmt.get('eps', 0) for stmt in income_statements_annual[:3]]

    # 全期間で正のEPS
    if not all(eps > 0 for eps in eps_values):
        return None

    # 毎年増加しているか確認
    for i in range(len(eps_values) - 1):
        if eps_values[i] <= eps_values[i + 1]:
            return None  # 増加していない年があれば除外

    # CAGR計算
    years = len(eps_values) - 1
    cagr = (pow(eps_values[0] / eps_values[-1], 1/years) - 1) * 100

    return cagr
```

#### オプション3: 売上成長率で代替（赤字企業向け）

```python
def calculate_revenue_growth_rate(income_statements_annual):
    """
    赤字企業向け: EPSの代わりに売上成長率を使用
    """
    if not income_statements_annual or len(income_statements_annual) < 3:
        return None

    revenue_values = [stmt.get('revenue', 0) for stmt in income_statements_annual[:3]]

    if revenue_values[0] > 0 and revenue_values[-1] > 0:
        years = len(revenue_values) - 1
        cagr = (pow(revenue_values[0] / revenue_values[-1], 1/years) - 1) * 100
        return cagr

    return None
```

### 7.2. 安定性スコアの計算改善

#### オプション1: 負のEPSを含むCV計算

```python
def calculate_stability_score_with_negatives(income_statements_quarterly):
    """
    負のEPSを含む変動係数を計算（絶対値ベース）
    """
    if len(income_statements_quarterly) < 8:
        return None

    eps_last_8q = [stmt.get('eps', 0) for stmt in income_statements_quarterly[:8]]
    eps_last_8q = [e for e in eps_last_8q if e is not None]

    if len(eps_last_8q) < 6:
        return None

    # 負のEPSを含む場合、絶対値の平均を使用
    eps_abs_mean = np.mean([abs(e) for e in eps_last_8q])
    eps_std = np.std(eps_last_8q)

    if eps_abs_mean > 0:
        cv = eps_std / eps_abs_mean

        # 負のEPSが含まれる場合、追加ペナルティ
        negative_count = sum(1 for e in eps_last_8q if e < 0)
        penalty = negative_count * 10  # 負のEPS 1期ごとに10点減点

        stability_score = max(0, 100 - (cv * 100) - penalty)
        return stability_score

    return None
```

#### オプション2: 正のEPSのみで計算（現在の方式を維持）

```python
def calculate_stability_score_positive_only(income_statements_quarterly):
    """
    正のEPSのみで変動係数を計算（現在の実装）
    """
    # 現在の実装を維持
    # ただし、条件を緩和: 8期中4期以上正のEPSがあればOK

    if len(income_statements_quarterly) < 8:
        return None

    eps_last_8q = [stmt.get('eps', 0) for stmt in income_statements_quarterly[:8]]
    eps_positive = [e for e in eps_last_8q if e is not None and e > 0]

    if len(eps_positive) >= 4:  # 6期 → 4期に緩和
        eps_mean = np.mean(eps_positive)
        eps_std = np.std(eps_positive)

        if eps_mean > 0:
            cv = eps_std / eps_mean
            stability_score = max(0, 100 - (cv * 100))
            return stability_score

    return None
```

### 7.3. 赤字企業の別指標

**提案**: 赤字企業専用の成長指標を追加

```python
def calculate_loss_company_metrics(income_statements_quarterly, income_statements_annual):
    """
    赤字企業向けの代替指標
    """
    metrics = {}

    # 1. 売上成長率（四半期）
    if income_statements_quarterly and len(income_statements_quarterly) >= 5:
        latest_revenue = income_statements_quarterly[0].get('revenue', 0)
        yoy_revenue = income_statements_quarterly[4].get('revenue', 0)

        if yoy_revenue > 0:
            revenue_growth = ((latest_revenue - yoy_revenue) / yoy_revenue) * 100
            metrics['quarterly_revenue_growth'] = revenue_growth

    # 2. 損失削減率（前年同期比で損失が縮小しているか）
    if income_statements_quarterly and len(income_statements_quarterly) >= 5:
        latest_eps = income_statements_quarterly[0].get('eps', 0)
        yoy_eps = income_statements_quarterly[4].get('eps', 0)

        if latest_eps < 0 and yoy_eps < 0:
            # 両方赤字の場合、損失の改善度を計算
            loss_improvement = ((yoy_eps - latest_eps) / abs(yoy_eps)) * 100
            metrics['loss_reduction_rate'] = loss_improvement

    # 3. 黒字転換の兆候（直近2四半期が黒字）
    if income_statements_quarterly and len(income_statements_quarterly) >= 2:
        recent_eps = [stmt.get('eps', 0) for stmt in income_statements_quarterly[:2]]
        if all(eps > 0 for eps in recent_eps):
            metrics['recently_profitable'] = True
        else:
            metrics['recently_profitable'] = False

    return metrics
```

---

## 8. 結論と推奨事項

### 8.1. IBDとCAN SLIMの違い

| 観点 | IBD EPS Rating | CAN SLIMスクリーニング |
|------|----------------|----------------------|
| **哲学** | 全銘柄にレーティング付与 | 優良成長株のみ抽出 |
| **赤字企業** | 低レーティングを付与 | 完全除外 |
| **過去の赤字** | ペナルティだが許容 | 除外基準 |
| **用途** | 相対的な評価・ランキング | スクリーニング・銘柄選択 |

### 8.2. 当プロジェクトの方向性

**現在の実装は「CAN SLIMスクリーニング寄り」（厳格）**

**推奨方針**:

#### オプションA: IBD方式に近づける（包括的）

- **メリット**: より多くの銘柄をカバー、ターンアラウンド株も評価可能
- **デメリット**: 低品質な銘柄も含まれる可能性
- **実装**: オプション1の改善策を採用

#### オプションB: CAN SLIM方式を維持（厳格）

- **メリット**: 高品質な成長株のみに絞り込める
- **デメリット**: 有望なターンアラウンド株を見逃す可能性
- **実装**: 現在の実装を維持、条件を明確化

#### オプションC: ハイブリッド方式（推奨）

1. **メイン指標**: CAN SLIM方式（厳格）
   - 年間成長率: 過去3年間毎年増加
   - 安定性スコア: 8期中6期以上正のEPS

2. **補完指標**: 赤字企業向け指標
   - 売上成長率
   - 損失削減率
   - 黒字転換フラグ

3. **フラグ付けシステム**:
   - `is_loss_company`: 赤字企業フラグ
   - `is_turnaround_candidate`: ターンアラウンド候補フラグ
   - `data_quality`: データ品質スコア

### 8.3. 実装の優先順位

#### Phase 1（短期）: 条件緩和と明確化

1. 年間成長率: 最新と最古が正なら計算（中間に赤字があってもOK）
2. 安定性スコア: 8期中4期以上正のEPSで計算（現在: 6期）
3. ドキュメント: 計算ロジックを明確化

#### Phase 2（中期）: 補完指標の追加

1. 売上成長率の計算
2. 損失削減率の計算
3. 赤字企業フラグの追加

#### Phase 3（長期）: 多段階レーティング

1. 優良成長株レーティング（CAN SLIM基準）
2. ターンアラウンド株レーティング（緩和基準）
3. 赤字成長株レーティング（売上ベース）

---

## 9. 参考文献

### 公式ドキュメント
- IBD Digital: EPS Rating - https://ibd.my.site.com/s/article/IBD-App-EPS-Rating
- O'Neil Proprietary Ratings and Rankings - https://origin.williamoneil.com/proprietary-ratings-and-rankings/

### スクリーニング実装
- AAII O'Neil's CAN SLIM Revised 3rd Edition Screen
- AAII: How to Use the CAN SLIM Approach to Screen for Growth Stocks

### 学術研究
- Minton and Schrand (1999) - Earnings Volatility Measurement
- Damodaran, A. - Valuing Firms with Negative Earnings (NYU Stern)

### コミュニティ実装
- GitHub: skyte/relative-strength (IBD RS Rating)
- Portfolio123: CAN SLIM Implementation
- AmiBroker Forum: IBD Relative Strength Discussion

---

**調査実施者**: Claude Code
**調査日**: 2025-11-14
**調査対象**: IBD EPS Rating, CAN SLIM Methodology, Community Implementations
