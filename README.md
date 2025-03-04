# レシート・クレジット履歴分析アプリケーション

## 概要
このアプリケーションは、レシートやクレジットカードの履歴を分析し、データを抽出するためのツールです。OCR技術を使用して画像からテキストを抽出し、Excelファイルとして出力することができます。家計簿や経費精算の効率化に役立ちます。

## 機能
- レシート画像のアップロードと解析
- クレジットカード履歴の分類
- データのExcelファイルへのエクスポート
- 勘定科目の分類と管理
- 日付、金額、店舗名などの自動抽出
- 複数のOCRエンジン（Tesseract OCR、EasyOCR）による高精度なテキスト認識

## 技術スタック
- Streamlit: Webアプリケーションフレームワーク
- Tesseract OCR: 画像からのテキスト抽出
- EasyOCR: 日本語テキスト認識
- pandas: データ処理
- openpyxl/xlsxwriter: Excelファイル生成
- OpenCV: 画像処理
- PIL (Pillow): 画像操作

## 依存関係
### Pythonライブラリ
```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
pytesseract>=0.3.10
opencv-python>=4.8.0
python-dotenv>=1.0.0
requests>=2.31.0
easyocr>=1.7.0
pyarrow>=10.0.1
pillow-heif>=0.15.0
openpyxl==3.1.2
xlsxwriter==3.1.0
```

### システムパッケージ
```
tesseract-ocr
libtesseract-dev
tesseract-ocr-jpn
tesseract-ocr-eng
libgl1
freeglut3-dev
libgtk2.0-dev
libsm6
libxext6
libxrender1
libglib2.0-0
poppler-utils
libleptonica-dev
libicu-dev
libpango1.0-dev
libcairo2-dev
```

## Streamlit Cloudでの実行
このアプリケーションはStreamlit Cloudで実行できます。以下の点に注意してください：

1. リポジトリのルートディレクトリとアプリケーションディレクトリの両方に`requirements.txt`ファイルが必要です。
2. 特に重要なライブラリは以下の通りです：
   - openpyxl==3.1.2
   - xlsxwriter==3.1.0
3. `packages.txt`ファイルには必要なシステムパッケージが含まれています。

## ローカルでの実行方法
### 環境のセットアップ
```bash
# リポジトリをクローン
git clone https://github.com/your-username/app_get_account_infomation.git
cd app_get_account_infomation

# 依存関係のインストール
pip install -r requirements.txt

# Tesseract OCRのインストール（プラットフォームによって異なります）
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract tesseract-lang
# Linux: apt-get install tesseract-ocr tesseract-ocr-jpn
```

### アプリケーションの実行
```bash
cd app_get_account_infomation
streamlit run app.py
```

## トラブルシューティング
### Excel出力の問題
Streamlit Cloud環境でExcelファイル出力に問題がある場合は、以下を確認してください：
- `requirements.txt`に`openpyxl==3.1.2`と`xlsxwriter==3.1.0`が含まれていることを確認
- リポジトリのルートディレクトリとアプリケーションディレクトリの両方に`requirements.txt`ファイルが存在することを確認

### OCR認識の問題
- Tesseract OCRが正しくインストールされていることを確認
- 日本語言語パックがインストールされていることを確認
- 画像の品質を向上させる（明るさ、コントラスト、解像度など）

## 貢献方法
1. リポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス
このプロジェクトはMITライセンスの下で公開されています。