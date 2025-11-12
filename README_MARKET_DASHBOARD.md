# Market Dashboard

マーケットダッシュボードは、Financial Modeling Prep APIを使用して、株式市場のデータを取得し、Googleスプレッドシートに視覚的に表示するPythonツールです。

## 機能

- **Market**: S&P 500、NASDAQ 100、Russell 2000などの主要インデックスの追跡
- **Sectors**: 各セクターETFのパフォーマンス追跡
- **Macro**: 米ドル指数、VIX、債券などのマクロ指標の追跡

各ティッカーについて以下の情報を表示：
- 現在価格と日次変化率
- 相対強度（RS）とRSパーセンタイル
- 各期間のパフォーマンス（YTD、1週間、1ヶ月、1年）
- 52週高値からの距離
- 移動平均との関係（10MA、20MA、50MA、200MA）

## セットアップ

### 1. 必要なパッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. Financial Modeling Prep APIキーの取得

1. [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs)にアクセス
2. アカウントを作成（無料プランあり）
3. APIキーを取得

### 3. Google Cloud Platform（GCP）のセットアップ

#### 3.1 プロジェクトの作成

1. [Google Cloud Console](https://console.cloud.google.com/)にアクセス
2. 新しいプロジェクトを作成

#### 3.2 Google Sheets APIとGoogle Drive APIの有効化

1. プロジェクトのダッシュボードで「APIとサービス」→「ライブラリ」を選択
2. 「Google Sheets API」を検索して有効化
3. 「Google Drive API」も検索して有効化

#### 3.3 サービスアカウントの作成

1. 「APIとサービス」→「認証情報」を選択
2. 「認証情報を作成」→「サービスアカウント」を選択
3. サービスアカウント名を入力（例: market-dashboard）
4. 「作成して続行」をクリック
5. ロールは設定不要（スキップ可能）
6. 「完了」をクリック

#### 3.4 認証情報（JSONキー）のダウンロード

1. 作成したサービスアカウントをクリック
2. 「キー」タブを選択
3. 「鍵を追加」→「新しい鍵を作成」を選択
4. 「JSON」を選択して「作成」をクリック
5. ダウンロードされたJSONファイルを`credentials.json`という名前でプロジェクトのルートディレクトリに保存

### 4. 環境変数の設定

```bash
# .env.exampleを.envにコピー
cp .env.example .env

# .envファイルを編集
nano .env
```

`.env`ファイルに以下の情報を設定：

```env
# FMP API Key
FMP_API_KEY=あなたのAPIキー

# Google Sheets Configuration
CREDENTIALS_FILE=credentials.json
SPREADSHEET_NAME=Market Dashboard
```

## 使い方

### 基本的な実行

```bash
python market_dashboard.py
```

### 初回実行時

1. スクリプトを実行すると、新しいGoogleスプレッドシートが自動的に作成されます
2. 作成されたスプレッドシートのURLがコンソールに表示されます
3. スプレッドシートを開くには、GCPで作成したサービスアカウントのメールアドレスと共有する必要があります

### スプレッドシートの共有

スプレッドシートを開くには、以下の手順で共有してください：

1. 作成されたスプレッドシートをGoogleドライブで開く
2. 「共有」をクリック
3. サービスアカウントのメールアドレス（`xxx@xxx.iam.gserviceaccount.com`）を追加
4. 「閲覧者」権限を付与

または、スクリプトが自動的にスプレッドシートを公開設定します（コード内で`self.spreadsheet.share()`を使用）。

## カスタマイズ

### ティッカーの追加・削除

`market_dashboard.py`の`main()`関数内で、以下の辞書を編集してティッカーを追加・削除できます：

```python
market_tickers = {
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ 100',
    # ここに追加
}
```

## トラブルシューティング

### APIキーエラー

```
エラー: FMP_API_KEYが設定されていません
```

→ `.env`ファイルにFMP APIキーを正しく設定してください。

### Google認証エラー

```
エラー: 認証情報ファイル 'credentials.json' が見つかりません
```

→ GCPでサービスアカウントを作成し、JSONキーをダウンロードして`credentials.json`として保存してください。

### スプレッドシートが見つからない

```
gspread.SpreadsheetNotFound
```

→ スプレッドシートがサービスアカウントと共有されていることを確認してください。

### API制限エラー

Financial Modeling Prepの無料プランには、API呼び出し回数に制限があります。スクリプトは各リクエスト間に0.4秒の待機時間を設けていますが、それでもエラーが発生する場合は、`time.sleep(0.4)`の値を増やしてください。

## 依存関係

- Python 3.8以上
- gspread >= 6.0.0
- google-auth >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- requests >= 2.31.0
- python-dotenv >= 1.0.0

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 注意事項

- Financial Modeling Prep APIの利用規約を遵守してください
- APIキーは秘密に保ち、.gitignoreに.envファイルを追加してください
- 無料プランの場合、API呼び出し回数に制限があります
