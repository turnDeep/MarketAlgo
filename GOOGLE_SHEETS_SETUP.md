# Google Sheets Setup Guide

このガイドでは、Market DashboardのデータをGoogleスプレッドシートに出力するための設定方法を説明します。

## 前提条件

- Googleアカウント
- Google Cloud Platformプロジェクト

## セットアップ手順

### 1. Google Cloud Platformでプロジェクトを作成

1. [Google Cloud Console](https://console.cloud.google.com/)にアクセス
2. 新しいプロジェクトを作成（例: "Market Dashboard"）
3. プロジェクトを選択

### 2. Google Sheets APIとGoogle Drive APIを有効化

1. 左側のメニューから「APIとサービス」→「ライブラリ」を選択
2. 「Google Sheets API」を検索して有効化
3. 「Google Drive API」を検索して有効化

### 3. サービスアカウントを作成

1. 「APIとサービス」→「認証情報」を選択
2. 「認証情報を作成」→「サービスアカウント」を選択
3. サービスアカウントの詳細を入力：
   - 名前: `market-dashboard-writer`
   - 説明: `Market Dashboard用スプレッドシート書き込みアカウント`
4. 「作成して続行」をクリック
5. ロールは設定不要（スキップ）
6. 「完了」をクリック

### 4. サービスアカウントキーを作成

1. 作成したサービスアカウントをクリック
2. 「キー」タブを選択
3. 「鍵を追加」→「新しい鍵を作成」
4. キーのタイプは「JSON」を選択
5. 「作成」をクリック
6. JSONファイルがダウンロードされます

### 5. 認証情報ファイルを配置

1. ダウンロードしたJSONファイルを`credentials.json`という名前でプロジェクトのルートディレクトリに配置

```bash
mv ~/Downloads/market-dashboard-xxxxx.json /path/to/MarketAlgo/credentials.json
```

### 6. Googleスプレッドシートを作成・共有

#### オプションA: 新規スプレッドシートを自動作成

プログラムが自動的にスプレッドシートを作成します。作成後、以下の手順でアクセス権限を設定してください：

1. プログラム実行後に表示されるスプレッドシートURLにアクセス
2. スプレッドシートを開く
3. 右上の「共有」ボタンをクリック
4. サービスアカウントのメールアドレス（`credentials.json`内の`client_email`）を追加
5. 権限を「編集者」に設定

#### オプションB: 既存のスプレッドシートを使用

1. Googleスプレッドシートで新しいスプレッドシートを作成
2. 右上の「共有」ボタンをクリック
3. サービスアカウントのメールアドレス（`credentials.json`内の`client_email`）を追加
4. 権限を「編集者」に設定
5. スプレッドシート名を環境変数`GOOGLE_SPREADSHEET_NAME`に設定

### 7. 環境変数の設定

`.env`ファイルを作成またはプロジェクトのルートディレクトリに配置し、以下の変数を設定：

```bash
# Google Sheets設定
GOOGLE_SHEETS_ENABLED=true
GOOGLE_CREDENTIALS_FILE=credentials.json
GOOGLE_SPREADSHEET_NAME=Market Dashboard

# FMP API Key（既存）
FMP_API_KEY=your_fmp_api_key_here
```

### 8. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

## 使用方法

### ダッシュボードの実行

```bash
python run_dashboard.py
```

プログラムは以下を実行します：
1. マーケットデータを収集
2. 分析を実行
3. JSONファイルに保存
4. HTMLダッシュボードを生成
5. Googleスプレッドシートにデータを書き込み（有効化されている場合）

### 出力されるシート

以下のシートが自動的に作成・更新されます：

1. **Summary** - 市場全体のサマリー
   - Market Exposure（スコア、レベル）
   - Factors vs SP500
   - Bond Yields
   - Power Trend

2. **Market Performance** - 主要指数のパフォーマンス
   - SPY, QQQ, IWM など

3. **Sectors Performance** - セクターETFのパフォーマンス
   - XLK, XLF, XLE など

4. **Macro Performance** - マクロ指標
   - VIX, TLT など

5. **Screener Results** - スクリーニング結果
   - Momentum 97
   - Explosive EPS Growth
   - Healthy Chart
   - Up on Volume
   - Top 2% RS Rating
   - 4% Bullish Yesterday

## トラブルシューティング

### エラー: "gspread is not installed"

```bash
pip install gspread google-auth
```

### エラー: "Failed to authenticate"

- `credentials.json`ファイルが正しい場所に配置されているか確認
- JSONファイルの内容が正しいか確認
- サービスアカウントキーが有効か確認

### エラー: "SpreadsheetNotFound"

- スプレッドシート名が正しいか確認
- サービスアカウントにスプレッドシートへのアクセス権限があるか確認

### エラー: "Insufficient Permission"

- サービスアカウントのメールアドレスをスプレッドシートに共有
- 権限を「編集者」に設定

## セキュリティに関する注意事項

⚠️ **重要**: `credentials.json`ファイルには機密情報が含まれています

- このファイルをGitにコミットしないでください
- `.gitignore`に`credentials.json`が追加されていることを確認してください
- サービスアカウントキーは定期的にローテーションすることを推奨します

## Dockerでの使用

Dockerコンテナ内でGoogleスプレッドシート機能を使用する場合：

1. `credentials.json`をコンテナにマウント
2. 環境変数を設定

docker-compose.yml例：

```yaml
services:
  market-dashboard:
    build: .
    volumes:
      - ./credentials.json:/app/credentials.json:ro
    environment:
      - GOOGLE_SHEETS_ENABLED=true
      - GOOGLE_CREDENTIALS_FILE=/app/credentials.json
      - GOOGLE_SPREADSHEET_NAME=Market Dashboard
      - FMP_API_KEY=${FMP_API_KEY}
```

## 参考リンク

- [Google Sheets API Documentation](https://developers.google.com/sheets/api)
- [gspread Documentation](https://docs.gspread.org/)
- [Google Cloud Console](https://console.cloud.google.com/)
