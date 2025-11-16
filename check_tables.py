"""
データベースのテーブル構造を確認
"""

import sqlite3


def check_tables():
    """データベースのテーブル一覧を表示"""
    conn = sqlite3.connect('ibd_data.db')
    cursor = conn.cursor()

    print("="*80)
    print("データベーステーブル一覧")
    print("="*80)

    # テーブル一覧を取得
    cursor.execute("""
        SELECT name, sql
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name
    """)

    tables = cursor.fetchall()

    if not tables:
        print("\n✗ データベースにテーブルが存在しません！")
        print("\n考えられる原因:")
        print("  1. データベースファイルが初期化されていない")
        print("  2. データ収集が実行されていない")
        print("  3. データベースファイルのパスが間違っている")
    else:
        print(f"\n見つかったテーブル数: {len(tables)}\n")

        for table_name, sql in tables:
            print(f"テーブル名: {table_name}")
            print(f"作成SQL: {sql}")
            print()

            # 各テーブルのレコード数を表示
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  → レコード数: {count}")
            print("-"*80)

    conn.close()


if __name__ == "__main__":
    check_tables()
