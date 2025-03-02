# レシート・クレジット履歴分析アプリ

このアプリケーションは、レシートやクレジットカードの履歴を分析し、支出を可視化するためのツールです。

https://appgetaccountinfomation-s8uvvk76lieq8hdugg3ehn.streamlit.app/

## 機能

- レシート画像からのテキスト抽出（OCR）
- クレジットカード履歴のCSVファイル解析
- 支出の分類と可視化
- 月次・カテゴリ別の支出分析

## 環境設定

### ローカル環境での実行

1. 必要なPythonパッケージをインストール:
   ```
   pip install -r requirements.txt
   ```

2. Tesseract OCRのインストール:
   - Windows: [Tesseract-OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki)からインストーラをダウンロード
   - Mac: `brew install tesseract tesseract-lang`
   - Linux: `sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-jpn tesseract-ocr-eng`

3. アプリケーションの実行:
   ```
   streamlit run app.py
   ```

### Streamlit Cloudでの実行

1. このリポジトリをStreamlit Cloudにデプロイします。
2. 必要なシステムパッケージは`packages.txt`に記載されています。
3. Pythonパッケージは`requirements.txt`に記載されています。

## トラブルシューティング

### OCR機能が動作しない場合

1. Tesseract OCRが正しくインストールされているか確認してください。
2. 環境変数`TESSERACT_PATH`にTesseractの実行ファイルへのパスを設定することで、検出を手動で上書きできます。
3. Streamlit Cloudでは、`packages.txt`に必要なパッケージが含まれていることを確認してください。

### デバッグモード

詳細なデバッグ情報を表示するには、環境変数`DEBUG_MODE`を`1`に設定してください。

```python
import os
os.environ['DEBUG_MODE'] = '1'
```

または、`.streamlit/config.toml`ファイルの`[global]`セクションで`DEBUG_MODE = "1"`のコメントを解除します。 