#!/usr/bin/env python3
"""
Google Sheets API接続テストスクリプト
"""

import gspread
import sys

def test_connection():
    """Google Sheets APIの接続をテスト"""

    print("=" * 70)
    print("Google Sheets API 接続テスト")
    print("=" * 70)

    # ステップ1: 認証情報の読み込み
    print("\n[ステップ1] 認証情報を読み込み中...")
    try:
        gc = gspread.service_account(filename='credentials.json')
        print("  ✓ 認証情報の読み込みに成功しました")
    except FileNotFoundError:
        print("  ✗ エラー: credentials.json が見つかりません")
        print("    → credentials.json をプロジェクトのルートディレクトリに配置してください")
        return False
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return False

    # ステップ2: サービスアカウント情報の表示
    print("\n[ステップ2] サービスアカウント情報:")
    import json
    with open('credentials.json', 'r') as f:
        creds = json.load(f)
        print(f"  プロジェクトID: {creds.get('project_id')}")
        print(f"  サービスアカウント: {creds.get('client_email')}")

    # ステップ3: スプレッドシートを開く
    print("\n[ステップ3] スプレッドシート 'Market Dashboard' を開いています...")
    spreadsheet_name = "Market Dashboard"

    try:
        spreadsheet = gc.open(spreadsheet_name)
        print(f"  ✓ スプレッドシート '{spreadsheet_name}' を開きました")
        print(f"    URL: https://docs.google.com/spreadsheets/d/{spreadsheet.id}")
    except gspread.SpreadsheetNotFound:
        print(f"  ✗ エラー: スプレッドシート '{spreadsheet_name}' が見つかりません")
        print("\n  解決方法:")
        print("  1. https://sheets.google.com で新しいスプレッドシートを作成")
        print(f"  2. タイトルを '{spreadsheet_name}' に変更")
        print(f"  3. サービスアカウント ({creds.get('client_email')}) を「編集者」として共有")
        return False
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return False

    # ステップ4: ワークシートの情報を取得
    print("\n[ステップ4] ワークシート情報:")
    try:
        worksheets = spreadsheet.worksheets()
        print(f"  ワークシート数: {len(worksheets)}")
        for ws in worksheets:
            print(f"    - {ws.title} ({ws.row_count} 行 x {ws.col_count} 列)")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return False

    # ステップ5: テストデータの書き込み
    print("\n[ステップ5] テストデータを書き込み中...")
    try:
        # 最初のワークシートを取得
        worksheet = spreadsheet.sheet1

        # A1セルに書き込み
        worksheet.update_acell('A1', 'Google Sheets API Test')
        worksheet.update_acell('A2', 'Status: Connected ✓')

        print("  ✓ テストデータの書き込みに成功しました")
        print(f"    → {spreadsheet_name} のA1, A2セルを確認してください")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        print("    → サービスアカウントの権限が「編集者」になっているか確認してください")
        return False

    # ステップ6: テストデータの読み込み
    print("\n[ステップ6] テストデータを読み込み中...")
    try:
        value = worksheet.acell('A1').value
        print(f"  ✓ A1セルの値: '{value}'")
    except Exception as e:
        print(f"  ✗ エラー: {e}")
        return False

    # 成功
    print("\n" + "=" * 70)
    print("✓ すべてのテストに成功しました！")
    print("Google Sheets API は正しく設定されています。")
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
