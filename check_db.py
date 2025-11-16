"""
データベースの内容を確認する最小限のスクリプト
"""

import sqlite3
from datetime import datetime


def check_database():
    """データベースの状態を確認"""
    conn = sqlite3.connect('ibd_data.db')
    cursor = conn.cursor()

    print("="*80)
    print("データベース確認")
    print("="*80)

    # 1. 価格データの件数
    print("\n1. 価格データの統計:")
    cursor.execute("SELECT COUNT(DISTINCT ticker) FROM price_history")
    ticker_count = cursor.fetchone()[0]
    print(f"   価格データがある銘柄数: {ticker_count}")

    cursor.execute("SELECT COUNT(*) FROM price_history")
    total_records = cursor.fetchone()[0]
    print(f"   価格データ総件数: {total_records}")

    # 2. SPYのデータ確認
    print("\n2. SPY (ベンチマーク) の価格データ:")
    cursor.execute("""
        SELECT COUNT(*) as days, MIN(date) as earliest, MAX(date) as latest
        FROM price_history
        WHERE ticker = 'SPY'
    """)
    result = cursor.fetchone()
    if result and result[0] > 0:
        print(f"   データ件数: {result[0]} 日分")
        print(f"   期間: {result[1]} ～ {result[2]}")

        # 最新5日分
        cursor.execute("""
            SELECT date, close
            FROM price_history
            WHERE ticker = 'SPY'
            ORDER BY date DESC
            LIMIT 5
        """)
        print("   最新5日分:")
        for row in cursor.fetchall():
            print(f"     {row[0]}: ${row[1]:.2f}")
    else:
        print("   ✗ SPYのデータが見つかりません！")

    # 3. サンプル銘柄のデータ確認
    print("\n3. サンプル銘柄の価格データ:")
    sample_tickers = ['CDTX', 'CELC', 'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']

    for ticker in sample_tickers:
        cursor.execute("""
            SELECT COUNT(*) as days, MIN(date) as earliest, MAX(date) as latest
            FROM price_history
            WHERE ticker = ?
        """, (ticker,))
        result = cursor.fetchone()

        if result and result[0] > 0:
            print(f"   {ticker}: {result[0]} 日分 ({result[1]} ～ {result[2]})")
        else:
            print(f"   {ticker}: データなし")

    # 4. RS値の計算状態
    print("\n4. RS値の計算状態:")
    cursor.execute("SELECT COUNT(*) FROM calculated_rs_values")
    rs_count = cursor.fetchone()[0]
    print(f"   RS値が計算済みの銘柄数: {rs_count}")

    if rs_count > 0:
        cursor.execute("""
            SELECT ticker, rs_value
            FROM calculated_rs_values
            LIMIT 10
        """)
        print("   サンプル銘柄のRS値:")
        for row in cursor.fetchall():
            print(f"     {row[0]}: {row[1]:.2f}")

    # 5. 日付マージのテスト
    print("\n5. 日付マージテスト (CDTX vs SPY):")
    cursor.execute("""
        SELECT COUNT(*) as common_dates
        FROM (
            SELECT DISTINCT date FROM price_history WHERE ticker = 'SPY'
        ) spy
        INNER JOIN (
            SELECT DISTINCT date FROM price_history WHERE ticker = 'CDTX'
        ) cdtx ON spy.date = cdtx.date
    """)
    common_dates = cursor.fetchone()[0]
    print(f"   共通する日付: {common_dates} 日")

    # 6. 最新のデータ収集日時を確認
    print("\n6. データ収集状況:")
    cursor.execute("""
        SELECT MAX(date) as latest_date
        FROM price_history
    """)
    latest = cursor.fetchone()[0]
    print(f"   最新データ日付: {latest}")

    conn.close()

    print("\n" + "="*80)


if __name__ == "__main__":
    check_database()
