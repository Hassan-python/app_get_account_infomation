# レシート・クレジット履歴分析アプリケーション

## 概要
このアプリケーションは、レシートやクレジットカードの履歴を分析し、データを抽出するためのツールです。OCR技術を使用して画像からテキストを抽出し、Excelファイルとして出力することができます。

## 機能
- レシート画像のアップロードと解析
- クレジットカード履歴の分析
- データのExcelファイルへのエクスポート
- 勘定科目の分類と管理

## 技術スタック
- Streamlit: Webアプリケーションフレームワーク
- Tesseract OCR: 画像からのテキスト抽出
- EasyOCR: 日本語テキスト認識
- pandas: データ処理
- openpyxl/xlsxwriter: Excelファイル生成

## Streamlit Cloudでの実行
このアプリケーションはStreamlit Cloudで実行できます。以下の点に注意してください：

1. `requirements.txt`ファイルには以下のライブラリが含まれています：
   - openpyxl==3.1.2
   - xlsxwriter==3.1.0

2. `packages.txt`ファイルには必要なシステムパッケージが含まれています：
   - tesseract-ocr
   - tesseract-ocr-jpn
   - その他の依存パッケージ

## ローカルでの実行方法
```bash
cd app_get_account_infomation
streamlit run app.py
```

## 注意事項
- Excelファイル出力機能を使用するには、openpyxlとxlsxwriterライブラリが必要です。
- 日本語OCRを使用するには、Tesseract OCRと日本語言語パックが必要です。
