# Google Sheets セットアップガイド

このガイドでは、IBDスクリーナーでGoogleスプレッドシートを使用する際のセットアップ方法と、よくある問題の解決方法を説明します。

## 🔧 セットアップ方法

### オプション1: 手動でスプレッドシートを作成（推奨）

最も簡単で確実な方法です。

1. **Googleスプレッドシートを作成**
   - https://sheets.google.com にアクセス
   - 「空白」をクリックして新しいスプレッドシートを作成
   - 左上のタイトルを「Market Dashboard」（または任意の名前）に変更

2. **サービスアカウントと共有**
   - `credentials.json` ファイルを開く
   - `client_email` の値をコピー（例: `my-service-account@project-id.iam.gserviceaccount.com`）
   - スプレッドシートの右上にある「共有」ボタンをクリック
   - コピーしたメールアドレスを貼り付け
   - 権限を「編集者」に設定
   - 「送信」をクリック

3. **完了**
   - スクリプトを実行すれば、このスプレッドシートが自動的に使用されます

### オプション2: 環境変数で自動作成（上級者向け）

`.env` ファイルに以下を追加することで、スプレッドシートの自動作成と所有権転送が可能です：

```bash
# .env ファイルに追加
GOOGLE_OWNER_EMAIL=your-email@gmail.com
```

**制限事項:**
- 同じドメイン内（例: 両方とも gmail.com）である必要があります
- Google Workspace アカウント間でも可能です
- 外部ドメインへの所有権転送はできません

## ❌ よくあるエラーと解決方法

### エラー: `APIError: [403]: The user's Drive storage quota has been exceeded`

**原因:**
- サービスアカウントはGoogle Driveのストレージクォータを持っていません
- スプレッドシートを作成しようとすると、サービスアカウントが所有者になりますが、ストレージがないため失敗します
- これはGoogleの仕様であり、実際のストレージが満杯かどうかとは関係ありません

**解決方法:**
上記の「オプション1: 手動でスプレッドシートを作成」を実行してください。

### エラー: `SpreadsheetNotFound`

**原因:**
- 指定された名前のスプレッドシートが見つからない
- または、サービスアカウントが共有されていない

**解決方法:**
1. スプレッドシート名が正しいか確認（デフォルト: "Market Dashboard"）
2. サービスアカウントのメールアドレスが共有リストに含まれているか確認
3. 権限が「編集者」または「オーナー」になっているか確認

### エラー: `Permission denied`

**原因:**
- サービスアカウントの権限が「閲覧者」になっている

**解決方法:**
- スプレッドシートの共有設定で、サービスアカウントの権限を「編集者」に変更

## 🔍 トラブルシューティング

### サービスアカウントのメールアドレスを確認する方法

```bash
# credentials.json から確認
cat credentials.json | grep client_email
```

または、Pythonで確認：

```python
import json

with open('credentials.json', 'r') as f:
    creds = json.load(f)
    print(f"サービスアカウント: {creds['client_email']}")
```

### スプレッドシート名を変更する方法

`.env` ファイルで変更できます：

```bash
SPREADSHEET_NAME=My Custom Dashboard
```

### 複数のスプレッドシートを使用する方法

環境変数を変更して実行：

```bash
SPREADSHEET_NAME="Dashboard 2" python run_ibd_screeners.py --run-screeners
```

## 📚 参考情報

- [Google Sheets API ドキュメント](https://developers.google.com/sheets/api)
- [gspread ドキュメント](https://docs.gspread.org/)
- [サービスアカウントについて](https://cloud.google.com/iam/docs/service-accounts)

## 💡 ベストプラクティス

1. **本番環境では、手動作成を推奨**
   - より確実で、デバッグが容易
   - 所有権が明確

2. **定期的にバックアップ**
   - Googleスプレッドシートの「ファイル」→「コピーを作成」

3. **サービスアカウントの認証情報は安全に管理**
   - `credentials.json` を `.gitignore` に追加
   - 決してGitHubにコミットしない

4. **アクセス権限は最小限に**
   - 必要なユーザーとサービスアカウントのみに共有
   - 不要になったアクセス権限は削除
