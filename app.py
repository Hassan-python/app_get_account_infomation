import streamlit as st
import requests
import pandas as pd
import json
import os
from io import BytesIO
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import datetime
import cv2
import numpy as np

# HEIC形式のサポートを追加
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False
    st.warning("HEICファイル形式のサポートには'pillow-heif'ライブラリが必要です。'pip install pillow-heif'でインストールしてください。")

from dotenv import load_dotenv

# 環境変数の読み込み
load_dotenv()

# DeepSeek APIキーの取得
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("DeepSeek APIキーが設定されていません。.envファイルにDEEPSEEK_API_KEYを設定してください。")

# 勘定科目のリスト
CATEGORIES = [
    "食費", 
    "交通費", 
    "日用品", 
    "衣料品", 
    "娯楽費", 
    "医療費", 
    "通信費", 
    "水道光熱費", 
    "住居費", 
    "その他"
]

# デフォルトの画像処理パラメータ
DEFAULT_PARAMS = {
    'detect_receipt': True,
    'enhance_contrast': True,
    'edge_detection_min': 50,
    'edge_detection_max': 150,
    'min_contour_area': 5000,
    'dilation_iterations': 2
}

def detect_receipt_area(img, params=None):
    """画像からレシート領域を検出する"""
    try:
        # 画像が空でないか確認
        if img is None or img.size == 0:
            st.error("レシート検出: 画像データが空です")
            return np.zeros((100, 100, 3), dtype=np.uint8)  # ダミー画像を返す
            
        # デフォルトパラメータ
        if params is None:
            params = DEFAULT_PARAMS.copy()
        else:
            # パラメータが渡された場合でも、デフォルト値で補完
            for key, value in DEFAULT_PARAMS.items():
                if key not in params:
                    params[key] = value
        
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去とエッジ強調
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cannyエッジ検出（パラメータ化）
        edge_min = params.get('edge_detection_min', 50)
        edge_max = params.get('edge_detection_max', 150)
        edges = cv2.Canny(blurred, edge_min, edge_max)
        
        # 膨張処理でエッジを強調（パラメータ化）
        dilation_iterations = params.get('dilation_iterations', 2)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilation_iterations)
        
        # 輪郭検出
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img  # 輪郭が見つからない場合は元の画像を返す
        
        # 面積でフィルタリング（パラメータ化）
        min_area = params.get('min_contour_area', 5000)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        if not filtered_contours:
            # 十分な大きさの輪郭がない場合は、元の画像に対して適応的二値化を試みる
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if not filtered_contours:
                return img  # それでも見つからない場合は元の画像を返す
        
        # 最大の輪郭を見つける（レシートと仮定）
        max_contour = max(filtered_contours, key=cv2.contourArea)
        
        # 輪郭の面積が小さすぎる場合は元の画像を返す
        if cv2.contourArea(max_contour) < min_area * 2:  # 閾値は調整可能
            return img
        
        # 輪郭を囲む最小の矩形を取得
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 矩形が画像全体に近い場合は元の画像を返す
        img_h, img_w = img.shape[:2]
        if w > 0.95 * img_w and h > 0.95 * img_h:
            return img
        
        # 余白を追加
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_w - x, w + 2 * padding)
        h = min(img_h - y, h + 2 * padding)
        
        # レシート領域を切り出す
        receipt_img = img[y:y+h, x:x+w]
        
        return receipt_img
    except Exception as e:
        st.error(f"レシート領域検出中にエラーが発生しました: {e}")
        return img  # エラーの場合は元の画像を返す

def apply_contrast_enhancement(img):
    """コントラスト強調を適用する"""
    # CLAHE（コントラスト制限適応ヒストグラム平坦化）を適用
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    return enhanced

def preprocess_image(image_bytes, params=None, is_credit=False):
    """OpenCVを使用して画像を前処理する（グレースケール変換とコントラスト強調）"""
    try:
        # デフォルトパラメータ
        if params is None:
            params = DEFAULT_PARAMS.copy()
        else:
            # パラメータが渡された場合でも、デフォルト値で補完
            for key, value in DEFAULT_PARAMS.items():
                if key not in params:
                    params[key] = value
        
        # まずPILで画像を開いてみる（HEIC形式などに対応するため）
        try:
            pil_img = Image.open(BytesIO(image_bytes))
            # 画像形式の情報を表示
            st.info(f"元の画像形式: {pil_img.format}, モード: {pil_img.mode}, サイズ: {pil_img.size}")
            
            # HEIC/HEIF形式の場合、PNG形式に変換して保存
            if pil_img.format == 'HEIC' or pil_img.format == 'HEIF':
                st.info(f"HEIC/HEIF形式の画像をPNG形式に変換しています...")
                # RGB形式に変換
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # 一時的にPNGとして保存
                temp_buffer = BytesIO()
                pil_img.save(temp_buffer, format="PNG")
                temp_buffer.seek(0)
                
                # 変換後の画像を表示
                st.image(temp_buffer, caption="PNG形式に変換された画像", width=300)
                
                # 変換後のバイトデータを使用
                image_bytes = temp_buffer.getvalue()
                
                # 変換後のPNG画像を再度開く
                pil_img = Image.open(BytesIO(image_bytes))
                st.info(f"変換後の画像形式: {pil_img.format}, モード: {pil_img.mode}")
                
                # OpenCVで処理するために変換
                img = np.array(pil_img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                # 通常のフォーマットはOpenCVで直接読み込む
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            st.warning(f"PILでの画像読み込みに失敗しました: {e}")
            # PILで開けない場合はOpenCVで試す
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 画像が正しく読み込まれたか確認
        if img is None:
            st.error("画像の読み込みに失敗しました。ファイル形式が対応していない可能性があります。")
            # ダミー画像を返す
            return Image.new('RGB', (100, 100), color = (255, 255, 255))
        
        # レシート領域の検出
        if params.get('detect_receipt', True):
            img = detect_receipt_area(img, params)
        
        # 検出したレシート画像を表示（デバッグ用）
        if (not is_credit and st.session_state.get('show_detected_receipt', True)) or \
           (is_credit and st.session_state.get('show_detected_receipt_credit', True)):
            detected_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_detected = Image.fromarray(detected_img)
            buffered = BytesIO()
            pil_detected.save(buffered, format="PNG")
            st.image(buffered.getvalue(), caption="検出されたレシート", width=300)
        
        # グレースケール変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # コントラスト強調を適用
        if params.get('enhance_contrast', True):
            gray = apply_contrast_enhancement(gray)
        
        # PIL形式に変換（pytesseractで使用するため）
        pil_img = Image.fromarray(gray)
        
        return pil_img
    except Exception as e:
        st.error(f"画像の前処理中にエラーが発生しました: {e}")
        # エラーの場合は元の画像をそのまま返す
        try:
            return Image.open(BytesIO(image_bytes))
        except Exception:
            # それでも開けない場合は空の画像を返す
            return Image.new('RGB', (100, 100), color = (255, 255, 255))

def process_image_with_ocr(image_bytes, params=None, is_credit=False):
    """画像からOCRを使用してテキストを抽出する"""
    try:
        # 画像形式の確認（デバッグ用）
        try:
            with Image.open(BytesIO(image_bytes)) as img:
                img_format = img.format
                img_mode = img.mode
                img_size = img.size
                st.info(f"画像情報: 形式={img_format}, モード={img_mode}, サイズ={img_size}")
        except Exception as e:
            st.warning(f"画像形式の確認中にエラーが発生しました: {e}")
        
        # 前処理を適用
        processed_image = preprocess_image(image_bytes, params, is_credit)
        
        # 前処理した画像を表示（デバッグ用）
        if (not is_credit and st.session_state.get('show_processed', True)) or \
           (is_credit and st.session_state.get('show_processed_credit', True)):
            buffered = BytesIO()
            processed_image.save(buffered, format="PNG")
            st.image(buffered.getvalue(), caption="前処理後の画像", width=300)
        
        # OCR処理
        # Tesseractのパラメータを設定して精度を向上
        # PSM値を調整（6: 単一のテキストブロックとして処理、3: 自動ページセグメンテーション）
        custom_config = r'--oem 3 --psm 3 -l jpn+eng'
        
        # OCR処理を実行
        try:
            text = pytesseract.image_to_string(processed_image, config=custom_config)
            
            # テキストが空の場合は別のPSM値で再試行
            if not text.strip():
                st.warning("最初のOCR処理でテキストが検出されませんでした。別の設定で再試行します。")
                # PSM値を変更して再試行
                alt_config = r'--oem 3 --psm 6 -l jpn+eng'
                text = pytesseract.image_to_string(processed_image, config=alt_config)
                
                # それでも空の場合は別の前処理を試す
                if not text.strip():
                    st.warning("2回目のOCR処理でもテキストが検出されませんでした。画像の二値化を試みます。")
                    # 画像を二値化して再試行
                    img_array = np.array(processed_image)
                    _, binary_img = cv2.threshold(img_array, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    binary_pil = Image.fromarray(binary_img)
                    
                    # 二値化した画像を表示
                    buffered = BytesIO()
                    binary_pil.save(buffered, format="PNG")
                    st.image(buffered.getvalue(), caption="二値化後の画像", width=300)
                    
                    # 二値化した画像でOCR実行
                    text = pytesseract.image_to_string(binary_pil, config=custom_config)
            
            return text
        except Exception as e:
            st.error(f"Tesseract OCR処理中にエラーが発生しました: {e}")
            return None
    except Exception as e:
        st.error(f"OCR処理中にエラーが発生しました: {e}")
        return None

def extract_receipt_info(text):
    """DeepSeek APIを使用してレシートから情報を抽出する"""
    if not DEEPSEEK_API_KEY:
        return None
    
    prompt = f"""
    以下のレシートのテキストから、店名、日付、合計金額を正確に抽出してJSON形式で出力してください。
    
    注意事項:
    - 日付はYYYY-MM-DD形式で出力してください。日付が不明確な場合は、空文字列を返してください。
    - 合計金額は数値のみを抽出してください。「合計」「小計」「総額」「お会計」などの表記を探してください。
    - 店名はレシートの最上部や最下部に記載されていることが多いです。
    - 抽出に自信がない場合は、該当フィールドを空文字列にしてください。
    
    テキスト:
    {text}
    
    出力形式:
    {{
        "store_name": "店名",
        "date": "YYYY-MM-DD",
        "total_amount": "金額"
    }}
    """
    
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        with st.spinner("レシート情報を抽出中..."):
            response = requests.post(url, headers=headers, json=data)
            
        if response.status_code == 200:
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            
            # JSONを抽出（余分なテキストがある場合に対応）
            try:
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    return data
                else:
                    st.warning("JSONデータが見つかりませんでした。")
                    return None
            except json.JSONDecodeError:
                st.warning("JSONの解析に失敗しました。")
                return None
        else:
            st.error(f"APIリクエストエラー: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"情報抽出中にエラーが発生しました: {e}")
        return None

def extract_credit_history(text):
    """DeepSeek APIを使用してクレジット履歴から情報を抽出する"""
    if not DEEPSEEK_API_KEY:
        return []
    
    prompt = f"""
    以下のクレジットカード履歴のテキストから、各取引の店名、日付、金額を正確に抽出してJSON形式で出力してください。
    
    注意事項:
    - 日付はYYYY-MM-DD形式で出力してください。日付が不明確な場合は、空文字列を返してください。
    - 金額は数値のみを抽出してください。
    - 複数の取引がある場合は、JSONオブジェクトのリストとして出力してください。
    - 抽出に自信がない場合は、該当フィールドを空文字列にしてください。
    
    テキスト:
    {text}
    
    出力形式:
    [
        {{
            "store_name": "店名1",
            "date": "YYYY-MM-DD",
            "total_amount": "金額1"
        }},
        {{
            "store_name": "店名2",
            "date": "YYYY-MM-DD",
            "total_amount": "金額2"
        }}
    ]
    """
    
    try:
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        with st.spinner("クレジット履歴情報を抽出中..."):
            response = requests.post(url, headers=headers, json=data)
            
        if response.status_code == 200:
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            
            # JSONを抽出（余分なテキストがある場合に対応）
            try:
                json_start = content.find("[")
                json_end = content.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                    return data
                else:
                    st.warning("JSONデータが見つかりませんでした。")
                    return []
            except json.JSONDecodeError:
                st.warning("JSONの解析に失敗しました。")
                return []
        else:
            st.error(f"APIリクエストエラー: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"情報抽出中にエラーが発生しました: {e}")
        return []

def validate_date(date_str):
    """日付の形式を検証し、未来の日付でないかチェックする"""
    try:
        # 空の文字列の場合
        if not date_str or date_str.strip() == "":
            return False, "日付が入力されていません。YYYY-MM-DD形式で入力してください。"
            
        # 形式の検証
        if len(date_str.split("-")) != 3:
            return False, "日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。"
            
        # 日付オブジェクトに変換
        date_obj = datetime.date.fromisoformat(date_str)
        today = datetime.date.today()
        
        # 未来の日付かチェック
        if date_obj > today:
            return False, f"未来の日付（{date_str}）が検出されました。正しい日付を入力してください。"
        
        # 極端に古い日付かチェック（例：10年以上前）
        ten_years_ago = today.replace(year=today.year - 10)
        if date_obj < ten_years_ago:
            return False, f"極端に古い日付（{date_str}）が検出されました。本当にこの日付で正しいですか？"
        
        # 年月日の妥当性チェック
        year, month, day = map(int, date_str.split("-"))
        
        # 年の範囲チェック
        if year < 2000 or year > today.year:
            return False, f"年（{year}）が有効な範囲外です。2000年から現在までの年を入力してください。"
        
        # 月の範囲チェック
        if month < 1 or month > 12:
            return False, f"月（{month}）が有効な範囲外です。1〜12の値を入力してください。"
        
        # 日の範囲チェック（月ごとの最大日数を考慮）
        days_in_month = [0, 31, 29 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 28, 
                         31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if day < 1 or day > days_in_month[month]:
            return False, f"{month}月の日（{day}）が有効な範囲外です。1〜{days_in_month[month]}の値を入力してください。"
        
        return True, None
    except ValueError as e:
        return False, f"日付の形式が正しくありません。YYYY-MM-DD形式で入力してください。エラー: {str(e)}"

def main():
    st.set_page_config(page_title="レシート・クレジット履歴分析", page_icon="📊", layout="wide")
    
    st.title("レシート・クレジット履歴分析アプリ")
    st.write("レシートやクレジットカード履歴の画像をアップロードして、情報を抽出し、Excelにまとめましょう。")
    
    # セッション状態の初期化
    if 'entries' not in st.session_state:
        st.session_state.entries = []
    
    # 画像データを保存するための辞書を初期化
    if 'image_data' not in st.session_state:
        st.session_state.image_data = {}
    
    # 表示設定の初期化
    if 'show_processed' not in st.session_state:
        st.session_state.show_processed = True
    if 'show_processed_credit' not in st.session_state:
        st.session_state.show_processed_credit = True
    if 'show_detected_receipt' not in st.session_state:
        st.session_state.show_detected_receipt = True
    if 'show_detected_receipt_credit' not in st.session_state:
        st.session_state.show_detected_receipt_credit = True
    
    # 勘定科目リストの初期化
    if 'categories' not in st.session_state:
        st.session_state.categories = CATEGORIES.copy()
    
    # タブの作成
    tab1, tab2, tab3 = st.tabs(["レシート処理", "クレジット履歴処理", "データ確認・ダウンロード"])
    
    with tab1:
        st.header("レシート画像のアップロード")
        receipt_files = st.file_uploader(
            "レシート画像をアップロードしてください", 
            type=["jpg", "jpeg", "png", "heic", "HEIC"], 
            accept_multiple_files=True,
            key="receipt_uploader"
        )
        
        # 前処理オプション
        st.subheader("画像表示・処理オプション")
        
        # 基本設定
        col1, col2, col3 = st.columns(3)
        with col1:
            # チェックボックスの値を直接セッション状態に代入するのではなく、戻り値を使用
            show_processed = st.checkbox("前処理後の画像を表示", value=st.session_state.show_processed, key="show_processed_checkbox")
            st.session_state.show_processed = show_processed
            
        with col2:
            show_detected = st.checkbox("検出されたレシートを表示", value=st.session_state.show_detected_receipt, key="show_detected_receipt_checkbox")
            st.session_state.show_detected_receipt = show_detected
            
        with col3:
            enhance_contrast = st.checkbox("コントラスト強調", value=True, key="enhance_contrast_checkbox")
        
        # 詳細設定（エキスパートモード）
        with st.expander("レシート検出の詳細設定"):
            st.write("レシート領域検出のパラメータを調整できます。")
            
            col1, col2 = st.columns(2)
            with col1:
                edge_min = st.slider("エッジ検出の最小閾値", min_value=10, max_value=200, value=50, step=10, key="edge_min_receipt")
                edge_max = st.slider("エッジ検出の最大閾値", min_value=50, max_value=300, value=150, step=10, key="edge_max_receipt")
            
            with col2:
                min_area = st.slider("最小輪郭面積", min_value=1000, max_value=20000, value=5000, step=1000, key="min_area_receipt")
                dilation_iter = st.slider("膨張処理の反復回数", min_value=1, max_value=5, value=2, step=1, key="dilation_iter_receipt")
        
        # 前処理パラメータを設定
        preprocessing_params = {
            'detect_receipt': True,
            'enhance_contrast': enhance_contrast,
            'edge_detection_min': edge_min,
            'edge_detection_max': edge_max,
            'min_contour_area': min_area,
            'dilation_iterations': dilation_iter
        }
        
        if st.button("レシート情報を抽出", key="extract_receipt"):
            if not receipt_files:
                st.warning("レシート画像をアップロードしてください。")
            else:
                for file in receipt_files:
                    # 元の画像の表示
                    st.image(file, caption=f"元画像: {file.name}", width=300)
                    
                    # 画像データを保存
                    file_bytes = file.read()
                    
                    # OCRでテキスト抽出
                    text = process_image_with_ocr(file_bytes, preprocessing_params)
                    if text:
                        # 抽出されたテキストを表示（デバッグ用）
                        with st.expander("抽出されたテキスト"):
                            st.text(text)
                            
                        # DeepSeek APIで情報抽出
                        entry = extract_receipt_info(text)
                        if entry:
                            entry["category"] = None
                            entry["source"] = "receipt"
                            entry["filename"] = file.name
                            
                            # 画像データをセッションに保存
                            image_id = f"receipt_{len(st.session_state.entries)}"
                            st.session_state.image_data[image_id] = file_bytes
                            entry["image_id"] = image_id
                            
                            st.session_state.entries.append(entry)
                            st.success(f"{file.name}からデータを抽出しました。")
                        else:
                            st.error(f"{file.name}からデータを抽出できませんでした。")
    
    with tab2:
        st.header("クレジット履歴画像のアップロード")
        credit_files = st.file_uploader(
            "クレジット履歴画像をアップロードしてください", 
            type=["jpg", "jpeg", "png", "heic", "HEIC"], 
            accept_multiple_files=True,
            key="credit_uploader"
        )
        
        # 前処理オプション
        st.subheader("画像表示・処理オプション")
        
        # 基本設定
        col1, col2, col3 = st.columns(3)
        with col1:
            # チェックボックスの値を直接セッション状態に代入するのではなく、戻り値を使用
            show_processed_credit = st.checkbox("前処理後の画像を表示", value=st.session_state.show_processed_credit, key="show_processed_credit_checkbox")
            st.session_state.show_processed_credit = show_processed_credit
            
        with col2:
            show_detected_credit = st.checkbox("検出されたレシートを表示", value=st.session_state.show_detected_receipt_credit, key="show_detected_receipt_credit_checkbox")
            st.session_state.show_detected_receipt_credit = show_detected_credit
            
        with col3:
            enhance_contrast_credit = st.checkbox("コントラスト強調", value=True, key="enhance_contrast_credit_checkbox")
        
        # 詳細設定（エキスパートモード）
        with st.expander("レシート検出の詳細設定"):
            st.write("レシート領域検出のパラメータを調整できます。")
            
            col1, col2 = st.columns(2)
            with col1:
                edge_min_credit = st.slider("エッジ検出の最小閾値", min_value=10, max_value=200, value=50, step=10, key="edge_min_credit")
                edge_max_credit = st.slider("エッジ検出の最大閾値", min_value=50, max_value=300, value=150, step=10, key="edge_max_credit")
            
            with col2:
                min_area_credit = st.slider("最小輪郭面積", min_value=1000, max_value=20000, value=5000, step=1000, key="min_area_credit")
                dilation_iter_credit = st.slider("膨張処理の反復回数", min_value=1, max_value=5, value=2, step=1, key="dilation_iter_credit")
        
        # 前処理パラメータを設定
        preprocessing_params_credit = {
            'detect_receipt': True,
            'enhance_contrast': enhance_contrast_credit,
            'edge_detection_min': edge_min_credit,
            'edge_detection_max': edge_max_credit,
            'min_contour_area': min_area_credit,
            'dilation_iterations': dilation_iter_credit
        }
        
        if st.button("クレジット履歴情報を抽出", key="extract_credit"):
            if not credit_files:
                st.warning("クレジット履歴画像をアップロードしてください。")
            else:
                for file in credit_files:
                    # 元の画像の表示
                    st.image(file, caption=f"元画像: {file.name}", width=300)
                    
                    # 画像データを保存
                    file_bytes = file.read()
                    
                    # OCRでテキスト抽出
                    text = process_image_with_ocr(file_bytes, preprocessing_params_credit, is_credit=True)
                    if text:
                        # 抽出されたテキストを表示（デバッグ用）
                        with st.expander("抽出されたテキスト"):
                            st.text(text)
                            
                        # DeepSeek APIで情報抽出
                        entries = extract_credit_history(text)
                        if entries:
                            for idx, entry in enumerate(entries):
                                entry["category"] = None
                                entry["source"] = "credit"
                                entry["filename"] = file.name
                                
                                # 画像データをセッションに保存
                                image_id = f"credit_{len(st.session_state.entries)}_{idx}"
                                st.session_state.image_data[image_id] = file_bytes
                                entry["image_id"] = image_id
                                
                                st.session_state.entries.append(entry)
                            st.success(f"{file.name}から{len(entries)}件のデータを抽出しました。")
                        else:
                            st.error(f"{file.name}からデータを抽出できませんでした。")
    
    with tab3:
        st.header("抽出されたデータ")
        
        # 勘定科目管理セクションを追加
        with st.expander("勘定科目の管理", expanded=False):
            st.subheader("勘定科目の管理")
            
            # 現在の勘定科目リストを表示
            st.write("現在の勘定科目リスト:")
            
            # 勘定科目の表示と削除ボタン
            categories_to_remove = []
            
            # 勘定科目を3列で表示
            cols = st.columns(3)
            for i, category in enumerate(st.session_state.categories):
                col_idx = i % 3
                with cols[col_idx]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"- {category}")
                    with col2:
                        if st.button("削除", key=f"delete_category_{i}"):
                            categories_to_remove.append(category)
            
            # 新規勘定科目の追加
            st.write("---")
            new_category = st.text_input("新規勘定科目", key="new_category_input")
            if st.button("勘定科目を追加", key="add_category_button"):
                if new_category:
                    if new_category in st.session_state.categories:
                        st.warning(f"勘定科目「{new_category}」は既に存在します。")
                    else:
                        st.session_state.categories.append(new_category)
                        st.success(f"勘定科目「{new_category}」を追加しました。")
                        st.rerun()
                else:
                    st.warning("勘定科目名を入力してください。")
            
            # 勘定科目の削除処理
            for category in categories_to_remove:
                if category in st.session_state.categories:
                    # この勘定科目を使用しているエントリがあるか確認
                    entries_using_category = [i for i, entry in enumerate(st.session_state.entries) if entry.get("category") == category]
                    
                    if entries_using_category:
                        st.warning(f"勘定科目「{category}」は{len(entries_using_category)}件のエントリで使用されています。削除できません。")
                    else:
                        st.session_state.categories.remove(category)
                        st.success(f"勘定科目「{category}」を削除しました。")
                        st.rerun()
        
        if not st.session_state.entries:
            st.info("まだデータが抽出されていません。レシートまたはクレジット履歴画像をアップロードして処理してください。")
        else:
            st.write(f"合計 {len(st.session_state.entries)} 件のデータが抽出されました。")
            
            # 無効なエントリを削除するためのリスト
            entries_to_remove = []
            
            # 各エントリに勘定科目を設定
            for i, entry in enumerate(st.session_state.entries):
                # 日付の検証
                date_is_valid = True
                date_error_msg = None
                if "date" in entry and entry["date"]:
                    date_is_valid, date_error_msg = validate_date(entry["date"])
                
                # エントリの表示と編集UI
                st.divider()
                
                # 日付が無効な場合は警告を表示
                if not date_is_valid:
                    st.warning(f"エントリ {i+1} ({entry.get('store_name', 'N/A')}): {date_error_msg}")
                
                # 2カラムレイアウト: 左側に画像、右側にデータ
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 対象の画像を表示
                    if "image_id" in entry and entry["image_id"] in st.session_state.image_data:
                        image_bytes = st.session_state.image_data[entry["image_id"]]
                        # 画像サイズを大きくし、画質を向上
                        try:
                            # PILで画像を開く
                            img = Image.open(BytesIO(image_bytes))
                            # 画像の解像度を保持したまま表示
                            st.image(img, caption=f"元画像: {entry.get('filename', 'N/A')}", width=300, use_container_width=True)
                            # 画像情報を表示
                            st.caption(f"画像サイズ: {img.width}x{img.height} ピクセル")
                        except Exception as e:
                            # エラーが発生した場合は元の方法で表示
                            st.image(image_bytes, caption=f"元画像: {entry.get('filename', 'N/A')}", width=250)
                            st.caption(f"画像表示エラー: {e}")
                
                with col2:
                    # データ編集用の3カラム
                    edit_col1, edit_col2, edit_col3 = st.columns([1, 1, 1])
                    
                    with edit_col1:
                        # 店名の表示と編集
                        store_name = st.text_input(
                            "店名",
                            value=entry.get('store_name', ''),
                            key=f"store_name_{i}"
                        )
                        st.session_state.entries[i]["store_name"] = store_name
                        
                        # 金額の表示と編集
                        total_amount = st.text_input(
                            "金額 (円)",
                            value=entry.get('total_amount', ''),
                            key=f"total_amount_{i}"
                        )
                        st.session_state.entries[i]["total_amount"] = total_amount
                    
                    with edit_col2:
                        # 日付の表示と編集
                        if date_is_valid:
                            corrected_date = st.text_input(
                                "日付 (YYYY-MM-DD)",
                                value=entry.get('date', ''),
                                key=f"date_{i}"
                            )
                            # 修正された日付を検証
                            is_valid, error_msg = validate_date(corrected_date)
                            if is_valid:
                                st.session_state.entries[i]["date"] = corrected_date
                            else:
                                # エラーメッセージをより目立つように表示
                                st.error(f"⚠️ 日付エラー: {error_msg}")
                                # 日付の背景色を変更するためのHTMLを使用
                                st.markdown(f"""
                                <div style="background-color: #ffcccc; padding: 10px; border-radius: 5px; margin-top: 5px;">
                                    <strong>無効な日付:</strong> {corrected_date}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            # 日付が無効な場合は編集フィールドを表示（より目立つように）
                            st.markdown(f"""
                            <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                <strong>⚠️ 日付エラー:</strong> {date_error_msg}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            corrected_date = st.text_input(
                                "**正しい日付を入力**（YYYY-MM-DD形式）:",
                                value=datetime.date.today().strftime("%Y-%m-%d"),
                                key=f"date_correction_{i}"
                            )
                            # 修正された日付を検証
                            is_valid, error_msg = validate_date(corrected_date)
                            if is_valid:
                                st.session_state.entries[i]["date"] = corrected_date
                                st.success("✅ 日付を更新しました")
                            else:
                                # エラーメッセージをより目立つように表示
                                st.error(f"⚠️ 日付エラー: {error_msg}")
                                # 日付の背景色を変更するためのHTMLを使用
                                st.markdown(f"""
                                <div style="background-color: #ffcccc; padding: 10px; border-radius: 5px; margin-top: 5px;">
                                    <strong>無効な日付:</strong> {corrected_date}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with edit_col3:
                        # 勘定科目のプルダウン
                        selected_category = st.selectbox(
                            "勘定科目",
                            options=st.session_state.categories,
                            index=0 if entry["category"] is None else 
                                  (st.session_state.categories.index(entry["category"]) 
                                   if entry["category"] in st.session_state.categories else 0),
                            key=f"category_{i}"
                        )
                        st.session_state.entries[i]["category"] = selected_category
                        
                        # エントリ削除ボタン
                        if st.button("このエントリを削除", key=f"remove_entry_{i}"):
                            entries_to_remove.append(i)
                        
                        # ファイル名とソースの表示
                        st.write(f"**ファイル**: {entry.get('filename', 'N/A')}")
                        st.write(f"**ソース**: {'レシート' if entry.get('source') == 'receipt' else 'クレジット履歴'}")
            
            # 無効なエントリを削除（逆順に削除して添字の問題を回避）
            for i in sorted(entries_to_remove, reverse=True):
                if i < len(st.session_state.entries):
                    # 画像データも削除
                    image_id = st.session_state.entries[i].get("image_id")
                    if image_id and image_id in st.session_state.image_data:
                        del st.session_state.image_data[image_id]
                    
                    # エントリを削除
                    st.session_state.entries.pop(i)
                    st.rerun()
            
            # Excelダウンロードボタン
            if st.button("Excelファイルを作成"):
                # カテゴリが選択されているか確認
                if any(entry["category"] is None for entry in st.session_state.entries):
                    st.warning("すべてのエントリに勘定科目を設定してください。")
                else:
                    # DataFrameの作成
                    df = pd.DataFrame(st.session_state.entries)
                    
                    # 不要なカラムを削除
                    if 'image_id' in df.columns:
                        df = df.drop(columns=['image_id'])
                    
                    # Excelファイルの作成
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        df.to_excel(writer, index=False, sheet_name="レシート・クレジット履歴")
                    
                    # ダウンロードボタン
                    st.download_button(
                        label="Excelファイルをダウンロード",
                        data=buffer.getvalue(),
                        file_name="receipt_credit_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            # データのクリアボタン
            if st.button("データをクリア"):
                st.session_state.entries = []
                st.session_state.image_data = {}
                st.rerun()

if __name__ == "__main__":
    main() 