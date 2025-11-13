#!/bin/bash
# FMP APIキー設定スクリプト

echo "=================================="
echo "FMP API Key 設定"
echo "=================================="
echo ""

# 現在のAPIキーを確認
if [ -f .env ]; then
    CURRENT_KEY=$(grep "^FMP_API_KEY=" .env | cut -d'=' -f2)
    echo "現在のAPIキー: $CURRENT_KEY"
    echo ""
fi

echo "新しいAPIキーを入力してください（キャンセルするにはEnter）:"
read -p "FMP_API_KEY: " NEW_KEY

if [ -z "$NEW_KEY" ]; then
    echo "キャンセルされました"
    exit 0
fi

# .envファイルを更新
if [ -f .env ]; then
    # 既存の.envファイルがある場合、FMP_API_KEYの行を置換
    sed -i "s/^FMP_API_KEY=.*/FMP_API_KEY=$NEW_KEY/" .env
    echo ""
    echo "✓ .envファイルを更新しました"
else
    # .envファイルがない場合、.env.exampleをコピーして更新
    if [ -f .env.example ]; then
        cp .env.example .env
        sed -i "s/^FMP_API_KEY=.*/FMP_API_KEY=$NEW_KEY/" .env
        echo ""
        echo "✓ .envファイルを作成しました"
    else
        echo "エラー: .env.exampleファイルが見つかりません"
        exit 1
    fi
fi

echo ""
echo "=================================="
echo "設定が完了しました！"
echo "=================================="
echo ""
echo "以下のコマンドでテストできます:"
echo "  python test_single_ticker.py"
echo ""
echo "または、本番を実行:"
echo "  python run_ibd_screeners.py"
