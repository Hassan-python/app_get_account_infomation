# ブランチ実験しています
import streamlit as st
# st.set_page_configはStreamlitの最初のコマンドとして実行する必要があります
st.set_page_config(page_title="レシート・クレジット履歴分析", page_icon="📊", layout="wide")

import requests
import pandas as pd
import json
import os
from io import BytesIO
from PIL import Image
import platform
import sys
import subprocess
import shutil
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# デバッグ情報を表示する関数
def debug_info(message):
    logger.info(message)
    if os.environ.get('DEBUG_MODE') == '1':
        st.info(f"[DEBUG] {message}")

# 環境情報を収集
debug_info(f"Platform: {platform.system()}")
debug_info(f"Python version: {sys.version}")
debug_info(f"Working directory: {os.getcwd()}")

# Streamlit Cloud環境の検出
is_streamlit_cloud = os.path.exists('/app')
if is_streamlit_cloud:
    st.info("🌩️ Streamlit Cloud環境で実行中です")
    debug_info("Streamlit Cloud環境を検出しました")
else:
    debug_info(f"ローカル環境で実行中: {platform.system()}")

# EasyOCRのインポートを試みる
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    st.success("✅ EasyOCRが正常にインポートされました。")
    
    # EasyOCRのリーダーを初期化（日本語と英語をサポート）
    @st.cache_resource
    def get_ocr_reader():
        try:
            reader = easyocr.Reader(['ja', 'en'])
            st.success("✅ EasyOCRリーダーが初期化されました。")
            return reader
        except Exception as e:
            st.error(f"❌ EasyOCRリーダーの初期化に失敗しました: {e}")
            return None
    
    # リーダーを取得
    ocr_reader = get_ocr_reader()
    
except ImportError:
    EASYOCR_AVAILABLE = False
    st.warning("⚠️ EasyOCRがインストールされていません。代わりにTesseractを使用します。")
    ocr_reader = None
    
    # Tesseractのインポートを試みる
    try:
        import pytesseract
        TESSERACT_AVAILABLE = True
    except ImportError:
        TESSERACT_AVAILABLE = False
        st.error("❌ pytesseractがインストールされていません。OCR機能は制限されます。")

# Tesseractのパス設定とエラーハンドリング（EasyOCRが利用できない場合のフォールバック）
if not EASYOCR_AVAILABLE:
    try:
        # 環境変数からTesseractパスを取得（優先）
        tesseract_path = os.environ.get('TESSERACT_PATH')
        if tesseract_path and os.path.exists(tesseract_path) and os.access(tesseract_path, os.X_OK):
            st.success(f"✅ 環境変数からTesseractパスを取得しました: {tesseract_path}")
            debug_info(f"環境変数から有効なTesseractパスを取得: {tesseract_path}")
        else:
            if tesseract_path:
                debug_info(f"環境変数のTesseractパスが無効です: {tesseract_path}")
            tesseract_path = None
            
            # Streamlit Cloud環境の処理
            if is_streamlit_cloud:
                st.info("🔍 Streamlit Cloud環境でTesseractを検索しています...")
                
                # システム情報を収集
                try:
                    debug_info("システム情報を収集中...")
                    os_info = subprocess.check_output("cat /etc/os-release 2>/dev/null || true", shell=True, text=True).strip()
                    debug_info(f"OS情報:\n{os_info}")
                    
                    # ディレクトリ構造を確認
                    dir_structure = subprocess.check_output("ls -la /app 2>/dev/null || true", shell=True, text=True).strip()
                    debug_info(f"/appディレクトリ構造:\n{dir_structure}")
                    
                    # .aptディレクトリの確認
                    if os.path.exists('/app/.apt'):
                        apt_structure = subprocess.check_output("ls -la /app/.apt 2>/dev/null || true", shell=True, text=True).strip()
                        debug_info(f"/app/.aptディレクトリ構造:\n{apt_structure}")
                except Exception as sys_err:
                    debug_info(f"システム情報収集中にエラー: {sys_err}")
                
                # パッケージの確認
                try:
                    # パッケージの確認
                    dpkg_cmd = "dpkg -l | grep -E 'tesseract|libtesseract' || true"
                    dpkg_output = subprocess.check_output(dpkg_cmd, shell=True, text=True).strip()
                    
                    if dpkg_output:
                        st.info(f"📦 インストールされているTesseract関連パッケージ:\n{dpkg_output}")
                        debug_info(f"Tesseract関連パッケージ:\n{dpkg_output}")
                    else:
                        st.warning("⚠️ Tesseract関連パッケージが見つかりません。packages.txtを確認してください。")
                        debug_info("Tesseract関連パッケージが見つかりません")
                except Exception as pkg_err:
                    debug_info(f"パッケージ情報の取得中にエラー: {pkg_err}")
                
                # Streamlit Cloudでの一般的なパスを試す
                cloud_paths = [
                    '/app/.apt/usr/bin/tesseract',
                    '/usr/bin/tesseract',
                    '/app/.apt/usr/share/tesseract-ocr/tesseract',
                    '/usr/share/tesseract-ocr/tesseract',
                    '/app/packages/tesseract-ocr/tesseract',
                    '/app/.apt/opt/tesseract/bin/tesseract',
                    '/app/.apt/usr/bin/tesseract-ocr',
                    '/app/.apt/usr/share/tesseract-ocr/4.00/tesseract',
                    '/app/.apt/usr/lib/tesseract-ocr/tesseract',
                    '/app/.apt/usr/lib/x86_64-linux-gnu/tesseract-ocr/tesseract'
                ]
                
                # パスの存在を確認
                for path in cloud_paths:
                    if os.path.exists(path):
                        if os.access(path, os.X_OK):
                            tesseract_path = path
                            st.success(f"✅ Tesseractパスが見つかりました: {path}")
                            debug_info(f"有効なTesseractパスを発見: {path}")
                            break
                        else:
                            debug_info(f"Tesseractパスが存在しますが、実行権限がありません: {path}")
                
                # which コマンドでパスを探す
                if not tesseract_path:
                    try:
                        which_cmd = "which tesseract 2>/dev/null || true"
                        which_output = subprocess.check_output(which_cmd, shell=True, text=True).strip()
                        
                        if which_output:
                            if os.path.exists(which_output) and os.access(which_output, os.X_OK):
                                tesseract_path = which_output
                                st.success(f"✅ whichコマンドでTesseractが見つかりました: {tesseract_path}")
                                debug_info(f"whichコマンドで有効なTesseractパスを発見: {which_output}")
                            else:
                                debug_info(f"whichコマンドで見つかったパスが無効です: {which_output}")
                    except Exception as which_err:
                        debug_info(f"whichコマンドでの検索中にエラー: {which_err}")
                
                # findコマンドで探す（最後の手段）
                if not tesseract_path:
                    try:
                        st.info("🔍 findコマンドでTesseractを検索しています...")
                        # /appディレクトリと/usrディレクトリを検索
                        find_cmd = "find /app /usr -name 'tesseract' -type f -executable 2>/dev/null || true"
                        find_output = subprocess.check_output(find_cmd, shell=True, text=True).strip()
                        
                        if find_output:
                            paths = [p for p in find_output.split('\n') if p.strip()]
                            if paths:
                                tesseract_path = paths[0]
                                st.success(f"✅ findコマンドでTesseractが見つかりました: {tesseract_path}")
                                debug_info(f"findコマンドで有効なTesseractパスを発見: {tesseract_path}")
                        else:
                            debug_info("findコマンドでTesseractが見つかりませんでした")
                    except Exception as find_err:
                        debug_info(f"findコマンドでの検索中にエラー: {find_err}")
                
                # Tesseract言語データの場所を確認
                try:
                    st.info("🔍 Tesseract言語データを検索しています...")
                    lang_cmd = "find /app /usr -path '*/tessdata' -type d 2>/dev/null || true"
                    lang_output = subprocess.check_output(lang_cmd, shell=True, text=True).strip()
                    
                    if lang_output:
                        lang_dirs = [p for p in lang_output.split('\n') if p.strip()]
                        if lang_dirs:
                            st.info(f"📁 Tesseract言語データディレクトリが見つかりました: {lang_dirs[0]}")
                            debug_info(f"言語データディレクトリ: {lang_dirs}")
                            
                            # 言語データの親ディレクトリにtesseractがあるか確認
                            for lang_dir in lang_dirs:
                                # 言語データディレクトリの内容を確認
                                try:
                                    lang_files = subprocess.check_output(f"ls -la {lang_dir} 2>/dev/null || true", shell=True, text=True).strip()
                                    debug_info(f"言語データディレクトリの内容 ({lang_dir}):\n{lang_files}")
                                    
                                    # jpnとengの言語ファイルが存在するか確認
                                    jpn_exists = os.path.exists(os.path.join(lang_dir, 'jpn.traineddata'))
                                    eng_exists = os.path.exists(os.path.join(lang_dir, 'eng.traineddata'))
                                    debug_info(f"言語ファイル: jpn={jpn_exists}, eng={eng_exists}")
                                    
                                    if not jpn_exists or not eng_exists:
                                        st.warning(f"⚠️ 言語ファイルが不足しています: jpn={jpn_exists}, eng={eng_exists}")
                                except Exception as ls_err:
                                    debug_info(f"言語ディレクトリの内容確認中にエラー: {ls_err}")
                                
                                # 言語データの親ディレクトリからtesseractを探す
                                parent_dir = os.path.dirname(lang_dir)
                                possible_paths = [
                                    os.path.join(parent_dir, 'tesseract'),
                                    os.path.join(parent_dir, 'bin', 'tesseract')
                                ]
                                
                                for possible_bin in possible_paths:
                                    if os.path.exists(possible_bin) and os.access(possible_bin, os.X_OK):
                                        if not tesseract_path:  # まだパスが見つかっていない場合のみ設定
                                            tesseract_path = possible_bin
                                            st.success(f"✅ 言語データから推測したTesseractパス: {tesseract_path}")
                                            debug_info(f"言語データから有効なTesseractパスを推測: {possible_bin}")
                                            break
                    else:
                        debug_info("Tesseract言語データディレクトリが見つかりませんでした")
                except Exception as lang_err:
                    debug_info(f"言語データの検索中にエラー: {lang_err}")
                
                # 最後の手段: 環境変数TESSDATA_PREFIXを確認
                if not tesseract_path and 'TESSDATA_PREFIX' in os.environ:
                    tessdata_prefix = os.environ['TESSDATA_PREFIX']
                    debug_info(f"TESSDATA_PREFIX環境変数が設定されています: {tessdata_prefix}")
                    
                    # TESSDATA_PREFIXの親ディレクトリにtesseractがあるか確認
                    parent_dir = os.path.dirname(tessdata_prefix)
                    possible_paths = [
                        os.path.join(parent_dir, 'tesseract'),
                        os.path.join(parent_dir, 'bin', 'tesseract')
                    ]
                    
                    for possible_bin in possible_paths:
                        if os.path.exists(possible_bin) and os.access(possible_bin, os.X_OK):
                            tesseract_path = possible_bin
                            st.success(f"✅ TESSDATA_PREFIXから推測したTesseractパス: {tesseract_path}")
                            debug_info(f"TESSDATA_PREFIXから有効なTesseractパスを推測: {possible_bin}")
                            break
            
            elif platform.system() == 'Windows':
                # Windowsの場合、一般的なインストールパスを試す
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                    r'C:\Users\Hassan\AppData\Local\Tesseract-OCR\tesseract.exe'
                ]
                
                # 存在するパスを探す
                for path in possible_paths:
                    if os.path.exists(path):
                        tesseract_path = path
                        st.success(f"✅ Tesseractパスが見つかりました: {path}")
                        debug_info(f"Windowsで有効なTesseractパスを発見: {path}")
                        break
            
            elif platform.system() == 'Linux':
                # 通常のLinux環境（Streamlit Cloud以外）
                linux_paths = [
                    '/usr/bin/tesseract',
                    '/usr/local/bin/tesseract',
                    '/usr/share/tesseract-ocr/tesseract'
                ]
                
                # パスの存在を確認
                for path in linux_paths:
                    if os.path.exists(path) and os.access(path, os.X_OK):
                        tesseract_path = path
                        st.success(f"✅ Tesseractパスが見つかりました: {path}")
                        debug_info(f"Linuxで有効なTesseractパスを発見: {path}")
                        break
                
                # shutil.whichを使用
                if not tesseract_path:
                    try:
                        tesseract_path = shutil.which('tesseract')
                        if tesseract_path:
                            st.success(f"✅ shutil.whichでTesseractが見つかりました: {tesseract_path}")
                            debug_info(f"shutil.whichで有効なTesseractパスを発見: {tesseract_path}")
                    except Exception as which_err:
                        debug_info(f"shutil.whichでの検索中にエラー: {which_err}")
        
        # Tesseractパスが見つかった場合の処理
        if tesseract_path:
            # パスが実際に存在するか最終確認
            if os.path.exists(tesseract_path) and os.access(tesseract_path, os.X_OK):
                # グローバル変数に設定
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
                st.success(f"✅ Tesseractパスを設定しました: {tesseract_path}")
                debug_info(f"Tesseractパスを設定: {tesseract_path}")
                
                # 環境変数にも設定（他のプロセスのため）
                os.environ['TESSERACT_PATH'] = tesseract_path
                
                # TESSDATA_PREFIXが設定されていない場合は推測して設定
                if 'TESSDATA_PREFIX' not in os.environ:
                    # tesseract_pathからTESSDATA_PREFIXを推測
                    tesseract_dir = os.path.dirname(tesseract_path)
                    possible_tessdata_dirs = [
                        os.path.join(tesseract_dir, 'tessdata'),
                        os.path.join(os.path.dirname(tesseract_dir), 'share', 'tesseract-ocr', 'tessdata'),
                        os.path.join(os.path.dirname(tesseract_dir), 'tessdata'),
                        '/usr/share/tesseract-ocr/tessdata',
                        '/app/.apt/usr/share/tesseract-ocr/tessdata'
                    ]
                    
                    for tessdata_dir in possible_tessdata_dirs:
                        if os.path.exists(tessdata_dir) and os.path.isdir(tessdata_dir):
                            os.environ['TESSDATA_PREFIX'] = tessdata_dir
                            debug_info(f"TESSDATA_PREFIX環境変数を設定: {tessdata_dir}")
                            break
                
                # Tesseractのバージョンを確認
                try:
                    version_output = subprocess.check_output([tesseract_path, '--version'], stderr=subprocess.STDOUT, text=True)
                    st.info(f"ℹ️ Tesseractバージョン情報: {version_output.splitlines()[0]}")
                    debug_info(f"Tesseractバージョン情報: {version_output}")
                    
                    # インストールされている言語パックを確認
                    try:
                        lang_output = subprocess.check_output([tesseract_path, '--list-langs'], stderr=subprocess.STDOUT, text=True)
                        st.info(f"ℹ️ 利用可能な言語: {lang_output}")
                        debug_info(f"利用可能な言語: {lang_output}")
                        
                        # 日本語と英語が利用可能か確認
                        if 'jpn' not in lang_output and 'eng' not in lang_output:
                            st.warning("⚠️ 日本語または英語の言語パックが見つかりません。OCR精度が低下する可能性があります。")
                            debug_info("日本語または英語の言語パックが見つかりません")
                    except Exception as lang_err:
                        st.warning(f"⚠️ 言語リストの取得に失敗しました: {lang_err}")
                        debug_info(f"言語リストの取得に失敗: {lang_err}")
                except Exception as ver_err:
                    st.warning(f"⚠️ Tesseractバージョンの確認に失敗しました: {ver_err}")
                    debug_info(f"Tesseractバージョンの確認に失敗: {ver_err}")
            else:
                st.error(f"❌ 見つかったTesseractパス {tesseract_path} は存在しないか、実行権限がありません。")
                debug_info(f"無効なTesseractパス: {tesseract_path}")
                tesseract_path = None
        
        # Tesseractが見つからない場合はダミー実装を使用
        if not tesseract_path:
            st.error("❌ Tesseractが見つかりません。OCR機能はダミー実装で代替します。")
            debug_info("Tesseractが見つからないため、ダミー実装を使用します")
            
            if is_streamlit_cloud:
                st.info("""
                📋 Streamlit Cloud環境では、packages.txtファイルに以下が含まれていることを確認してください：
                ```
                tesseract-ocr
                libtesseract-dev
                tesseract-ocr-jpn
                tesseract-ocr-eng
                ```
                """)
                
                # packages.txtの内容を確認
                try:
                    if os.path.exists('packages.txt'):
                        with open('packages.txt', 'r') as f:
                            packages_content = f.read()
                        debug_info(f"packages.txtの内容:\n{packages_content}")
                    else:
                        debug_info("packages.txtファイルが見つかりません")
                except Exception as pkg_err:
                    debug_info(f"packages.txtの読み込み中にエラー: {pkg_err}")
            else:
                st.info("ℹ️ Tesseractのインストール方法: https://github.com/UB-Mannheim/tesseract/wiki")
            
            # ダミーのOCR機能を提供
            class DummyTesseract:
                @staticmethod
                def image_to_string(image, **kwargs):
                    return "OCR機能が利用できません。Tesseractがインストールされていないか、正しく設定されていません。"
            
            # グローバル名前空間に追加
            pytesseract.image_to_string = DummyTesseract.image_to_string
            TESSERACT_AVAILABLE = False
    except Exception as e:
        st.error(f"❌ Tesseract設定中にエラーが発生しました: {e}")
        debug_info(f"Tesseract設定中にエラー: {e}")
        
        # エラーが発生した場合はダミー実装を使用
        try:
            class DummyTesseract:
                @staticmethod
                def image_to_string(image, **kwargs):
                    return "OCR機能が利用できません。Tesseractの設定中にエラーが発生しました。"
            
            # グローバル名前空間に追加
            pytesseract.image_to_string = DummyTesseract.image_to_string
            TESSERACT_AVAILABLE = False
        except Exception:
            st.error("❌ ダミーTesseractの設定にも失敗しました。OCR機能は完全に無効化されます。")
            debug_info("ダミーTesseractの設定にも失敗")
            TESSERACT_AVAILABLE = False

import datetime
import numpy as np

# cv2のインポートを試みる - Python 3.12対応の堅牢なエラーハンドリング
try:
    import cv2
    st.success(f"OpenCV (cv2) モジュールが正常にインポートされました。バージョン: {cv2.__version__}")
    CV2_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV (cv2) モジュールのインポートに失敗しました: {e}")
    # 詳細なエラー情報を表示
    st.error(f"Python バージョン: {sys.version}")
    st.error("インストールされているパッケージを確認します...")
    
    # インストールされているパッケージの情報を表示
    try:
        import pkg_resources
        installed_packages = [f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]
        opencv_packages = [pkg for pkg in installed_packages if "opencv" in pkg.lower()]
        if opencv_packages:
            st.error(f"インストールされているOpenCVパッケージ: {', '.join(opencv_packages)}")
        else:
            st.error("OpenCV関連のパッケージが見つかりません。")
    except Exception as pkg_err:
        st.error(f"パッケージ情報の取得に失敗しました: {pkg_err}")
        
    # ダミーのcv2モジュールを作成
    class DummyCV2:
        def __init__(self):
            self.COLOR_BGR2GRAY = 6
            self.THRESH_BINARY = 0
            self.THRESH_BINARY_INV = 1
            self.RETR_EXTERNAL = 0
            self.CHAIN_APPROX_SIMPLE = 1
            self.ADAPTIVE_THRESH_GAUSSIAN_C = 1
            self.__version__ = "dummy"
            
        def cvtColor(self, img, code):
            # グレースケール変換のダミー実装
            if code == self.COLOR_BGR2GRAY and len(img.shape) == 3:
                return np.mean(img, axis=2).astype(np.uint8)
            return img
            
        def threshold(self, img, thresh, maxval, type):
            # 二値化のダミー実装
            if type == self.THRESH_BINARY:
                return 1, (img > thresh) * maxval
            elif type == self.THRESH_BINARY_INV:
                return 1, (img <= thresh) * maxval
            return 1, img
            
        def findContours(self, img, mode, method):
            # 輪郭検出のダミー実装
            return [], None
            
        def GaussianBlur(self, img, ksize, sigmaX):
            # ぼかしのダミー実装
            return img
            
        def resize(self, img, dsize, fx=None, fy=None, interpolation=None):
            # リサイズのダミー実装
            if dsize is not None:
                h, w = dsize
                return np.zeros((h, w) if len(img.shape) == 2 else (h, w, img.shape[2]), dtype=img.dtype)
            return img
            
        def adaptiveThreshold(self, img, maxValue, adaptiveMethod, thresholdType, blockSize, C):
            # 適応的二値化のダミー実装
            return np.ones_like(img) * maxValue
            
        def dilate(self, img, kernel, iterations=1):
            # 膨張処理のダミー実装
            return img
            
        def Canny(self, img, threshold1, threshold2):
            # エッジ検出のダミー実装
            return np.zeros_like(img)
            
        def boundingRect(self, contour):
            # 矩形取得のダミー実装
            return 0, 0, 100, 100
            
        def contourArea(self, contour):
            # 輪郭面積のダミー実装
            return 0
            
        def createCLAHE(self, clipLimit=2.0, tileGridSize=(8, 8)):
            # CLAHEのダミー実装
            class DummyCLAHE:
                def apply(self, img):
                    return img
            return DummyCLAHE()
            
        def imdecode(self, buf, flags):
            # 画像デコードのダミー実装
            try:
                from PIL import Image
                import numpy as np
                img = Image.open(BytesIO(buf))
                return np.array(img)
            except:
                return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # ダミーのcv2モジュールをグローバル名前空間に追加
    cv2 = DummyCV2()
    CV2_AVAILABLE = False
    st.warning("OpenCV (cv2) モジュールが見つかりませんでした。一部の画像処理機能が制限されます。")

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
    st.warning("DeepSeek APIキーが設定されていません。OCR結果からの情報抽出機能は制限されます。")
    st.info("情報抽出機能を使用するには、.envファイルにDEEPSEEK_API_KEYを設定してください。")

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
        
        # EasyOCRが利用可能な場合はEasyOCRを使用
        if EASYOCR_AVAILABLE and ocr_reader:
            try:
                # PIL画像をnumpy配列に変換
                img_array = np.array(processed_image)
                
                # EasyOCRでテキスト認識
                with st.spinner("EasyOCRでテキスト認識中..."):
                    results = ocr_reader.readtext(img_array)
                
                # 結果を連結してテキストを生成
                if results:
                    text = "\n".join([result[1] for result in results])
                    return text
                else:
                    st.warning("EasyOCRでテキストが検出されませんでした。")
                    return "テキストが検出されませんでした。手動でデータを入力してください。"
            except Exception as e:
                st.error(f"EasyOCR処理中にエラーが発生しました: {e}")
                st.warning("TesseractにフォールバックしてOCR処理を試みます。")
                # EasyOCRが失敗した場合はTesseractにフォールバック
        
        # Tesseractを使用（EasyOCRが利用できない場合や失敗した場合）
        if TESSERACT_AVAILABLE:
            # OCR処理
            # Tesseractのパラメータを設定して精度を向上
            # PSM値を調整（6: 単一のテキストブロックとして処理、3: 自動ページセグメンテーション）
            custom_config = r'--oem 3 --psm 3 -l jpn+eng'
            
            # OCR処理を実行
            try:
                # OCR実行
                text = pytesseract.image_to_string(processed_image, config=custom_config)
                
                # テキストが空の場合は別のPSM値で再試行
                if not text.strip() or text.strip() == "OCR機能が利用できません。Tesseractがインストールされていないか、正しく設定されていません。":
                    if text.strip() == "OCR機能が利用できません。Tesseractがインストールされていないか、正しく設定されていません。":
                        st.warning("Tesseractが利用できないため、OCR処理をスキップします。")
                        return "OCR機能が利用できません。手動でデータを入力してください。"
                    
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
                return "OCR処理中にエラーが発生しました。手動でデータを入力してください。"
        else:
            st.error("OCR機能が利用できません。EasyOCRもTesseractも利用できません。")
            return "OCR機能が利用できません。手動でデータを入力してください。"
    except Exception as e:
        st.error(f"OCR処理中にエラーが発生しました: {e}")
        return "OCR処理中にエラーが発生しました。手動でデータを入力してください。"

def extract_receipt_info(text):
    """DeepSeek APIを使用してレシートから情報を抽出する"""
    if not DEEPSEEK_API_KEY:
        st.warning("DeepSeek APIキーが設定されていないため、情報抽出機能は利用できません。")
        # APIキーがない場合は空のデータを返す
        return {
            "store_name": "",
            "date": "",
            "total_amount": ""
        }
    
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
        st.warning("DeepSeek APIキーが設定されていないため、情報抽出機能は利用できません。")
        # APIキーがない場合は空のリストを返す
        return [{
            "store_name": "",
            "date": "",
            "total_amount": ""
        }]
    
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
    st.title("レシート・クレジット履歴分析アプリ")
    st.write("レシートやクレジットカード履歴の画像をアップロードして、情報を抽出し、Excelにまとめましょう。")
    
    # Excelライブラリのインポート確認
    try:
        import openpyxl
        OPENPYXL_AVAILABLE = True
        debug_info(f"✅ openpyxlが正常にインポートされました。バージョン: {openpyxl.__version__}")
    except ImportError:
        OPENPYXL_AVAILABLE = False
        st.warning("⚠️ openpyxlをインポートできませんでした。Excel出力機能が制限される可能性があります。")

    try:
        import xlsxwriter
        XLSXWRITER_AVAILABLE = True
        debug_info(f"✅ xlsxwriterが正常にインポートされました。バージョン: {xlsxwriter.__version__}")
    except ImportError:
        XLSXWRITER_AVAILABLE = False
        st.warning("⚠️ xlsxwriterをインポートできませんでした。Excel出力機能が制限される可能性があります。")

    # Excelエンジンの状態をまとめて表示
    excel_status = []
    if 'openpyxl' in sys.modules:
        excel_status.append("openpyxl: ✓")
    else:
        excel_status.append("openpyxl: ✗")
    
    if 'xlsxwriter' in sys.modules:
        excel_status.append("xlsxwriter: ✓")
    else:
        excel_status.append("xlsxwriter: ✗")
    
    if 'openpyxl' in sys.modules or 'xlsxwriter' in sys.modules:
        st.success(f"✅ Excel出力機能が利用可能です。({' | '.join(excel_status)})")
    else:
        st.error("❌ Excel出力機能が利用できません。requirements.txtにopenpyxlとxlsxwriterが含まれていることを確認してください。")
    
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
    
    # 初期化メッセージを保存するためのリスト
    initialization_messages = []
    
    # 初期化メッセージを収集する関数
    def collect_message(message, type="info"):
        if type == "success":
            initialization_messages.append(("success", message))
        elif type == "warning":
            initialization_messages.append(("warning", message))
        elif type == "error":
            initialization_messages.append(("error", message))
        else:
            initialization_messages.append(("info", message))
    
    # 元のst.success、st.warning、st.errorなどの関数を一時的に保存
    original_success = st.success
    original_warning = st.warning
    original_error = st.error
    original_info = st.info
    
    # 一時的に関数をオーバーライド
    st.success = lambda msg: collect_message(msg, "success")
    st.warning = lambda msg: collect_message(msg, "warning")
    st.error = lambda msg: collect_message(msg, "error")
    st.info = lambda msg: collect_message(msg, "info")
    
    # タブの作成
    tab1, tab2, tab3 = st.tabs(["レシート処理", "クレジット履歴処理", "データ確認・ダウンロード"])
    
    # 関数を元に戻す
    st.success = original_success
    st.warning = original_warning
    st.error = original_error
    st.info = original_info
    
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
                    
                    # Excelファイルの作成（BytesIOを使用）
                    buffer = BytesIO()
                    excel_created = False
                    
                    # インポート状態に基づいて最適なエンジンを選択
                    if 'openpyxl' in sys.modules:
                        try:
                            # openpyxlエンジンを使用
                            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                                df.to_excel(writer, index=False, sheet_name="レシート・クレジット履歴")
                            excel_created = True
                            st.success("✅ openpyxlエンジンでExcelファイルを作成しました。")
                            debug_info("openpyxlエンジンでExcelファイルを作成しました。")
                        except Exception as e:
                            st.error(f"❌ openpyxlでのExcelファイル作成中にエラーが発生しました: {str(e)}")
                            debug_info(f"openpyxlエラー: {str(e)}")
                    
                    # openpyxlが失敗した場合はxlsxwriterを試す
                    if not excel_created and 'xlsxwriter' in sys.modules:
                        try:
                            # xlsxwriterエンジンを使用
                            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                                df.to_excel(writer, index=False, sheet_name="レシート・クレジット履歴")
                            excel_created = True
                            st.success("✅ xlsxwriterエンジンでExcelファイルを作成しました。")
                            debug_info("xlsxwriterエンジンでExcelファイルを作成しました。")
                        except Exception as e:
                            st.error(f"❌ xlsxwriterでのExcelファイル作成中にエラーが発生しました: {str(e)}")
                            debug_info(f"xlsxwriterエラー: {str(e)}")
                    
                    # 両方が失敗した場合はデフォルトエンジンを試す
                    if not excel_created:
                        try:
                            # デフォルトエンジンを使用
                            df.to_excel(buffer, index=False)
                            excel_created = True
                            st.success("✅ デフォルトエンジンでExcelファイルを作成しました。")
                            debug_info("デフォルトエンジンでExcelファイルを作成しました。")
                        except Exception as e:
                            st.error(f"❌ Excelファイルの作成に失敗しました: {str(e)}")
                            debug_info(f"デフォルトエンジンエラー: {str(e)}")
                            return
                    
                    # ダウンロードボタン（MIMEタイプを正確に指定）
                    if excel_created:
                        st.download_button(
                            label="Excelファイルをダウンロード",
                            data=buffer.getvalue(),
                            file_name="receipt_credit_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.error("Excelファイルの作成に失敗しました。")
            
            # データのクリアボタン
            if st.button("データをクリア"):
                st.session_state.entries = []
                st.session_state.image_data = {}
                st.rerun()

if __name__ == "__main__":
    main() 