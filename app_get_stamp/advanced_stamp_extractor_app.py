import streamlit as st
import cv2
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="高度な押印抽出アプリ", layout="wide")

def extract_stamps_kmeans(img, min_area=100, blur_kernel=(5, 5)):
    """
    K-meansクラスタリングを使用して画像から押印部分を抽出する関数
    
    Parameters:
    -----------
    img : numpy.ndarray
        入力画像（BGR形式）
    min_area : int
        抽出する押印の最小面積
    blur_kernel : tuple
        ぼかし処理のカーネルサイズ
        
    Returns:
    --------
    list
        抽出された押印画像のリスト
    numpy.ndarray
        マスク画像
    list
        各押印の座標情報 (x, y, w, h)
    """
    if img is None:
        st.error("画像を読み込めません。")
        return [], None, []
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # K-meansクラスタリング
    pixels = blurred.ravel().reshape(-1, 1).astype(np.float32)
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(pixels)
    labels = kmeans.predict(pixels)
    
    # クラスタのカウント
    count0, count1 = np.sum(labels == 0), np.sum(labels == 1)
    
    # 少数派のクラスタを押印と仮定
    mask = (labels == 0).reshape(gray.shape) if count0 < count1 else (labels == 1).reshape(gray.shape)
    
    # 連結成分の抽出
    return process_mask(img, mask, min_area)

def extract_stamps_color(img, color_range='red', min_area=100, blur_kernel=(5, 5)):
    """
    色空間を使用して特定の色の押印を抽出する関数
    
    Parameters:
    -----------
    img : numpy.ndarray
        入力画像（BGR形式）
    color_range : str
        抽出する色範囲 ('red', 'blue', 'black')
    min_area : int
        抽出する押印の最小面積
    blur_kernel : tuple
        ぼかし処理のカーネルサイズ
        
    Returns:
    --------
    list
        抽出された押印画像のリスト
    numpy.ndarray
        マスク画像
    list
        各押印の座標情報 (x, y, w, h)
    """
    if img is None:
        st.error("画像を読み込めません。")
        return [], None, []
    
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 色範囲の定義
    if color_range == 'red':
        # 赤色は色相環の両端にあるため、2つのマスクを作成して結合
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    elif color_range == 'blue':
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    elif color_range == 'black':
        # 黒色は彩度と明度が低い
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 100, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)
    
    else:  # デフォルトは赤
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
    
    # ノイズ除去
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 連結成分の抽出
    return process_mask(img, mask, min_area)

def extract_stamps_edge(img, min_area=100, blur_kernel=(5, 5), canny_low=50, canny_high=150):
    """
    エッジ検出を使用して押印部分を抽出する関数
    
    Parameters:
    -----------
    img : numpy.ndarray
        入力画像（BGR形式）
    min_area : int
        抽出する押印の最小面積
    blur_kernel : tuple
        ぼかし処理のカーネルサイズ
    canny_low : int
        Cannyエッジ検出の低閾値
    canny_high : int
        Cannyエッジ検出の高閾値
        
    Returns:
    --------
    list
        抽出された押印画像のリスト
    numpy.ndarray
        マスク画像
    list
        各押印の座標情報 (x, y, w, h)
    """
    if img is None:
        st.error("画像を読み込めません。")
        return [], None, []
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    # Cannyエッジ検出
    edges = cv2.Canny(blurred, canny_low, canny_high)
    
    # 輪郭を閉じるためのモルフォロジー演算
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 連結成分の抽出
    return process_mask(img, closed_edges, min_area)

def process_mask(img, mask, min_area):
    """
    マスク画像から連結成分を抽出し、押印を特定する関数
    
    Parameters:
    -----------
    img : numpy.ndarray
        元の画像
    mask : numpy.ndarray
        二値マスク画像
    min_area : int
        抽出する押印の最小面積
        
    Returns:
    --------
    list
        抽出された押印画像のリスト
    numpy.ndarray
        マスク画像
    list
        各押印の座標情報 (x, y, w, h)
    """
    # 連結成分の抽出
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    stamps = []
    stamp_coords = []
    
    # 各連結成分を処理
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            stamp_img = img[y:y+h, x:x+w]
            stamps.append(stamp_img)
            stamp_coords.append((x, y, w, h))
    
    return stamps, mask, stamp_coords

def main():
    st.title("高度な押印抽出アプリ")
    
    st.markdown("""
    このアプリは画像から押印部分を自動的に抽出します。複数の抽出方法を試すことができます。
    
    ### 使い方
    1. 左側のサイドバーでパラメータと抽出方法を選択します
    2. 画像をアップロードします
    3. 抽出結果を確認します
    
    ### 抽出方法
    - **K-meansクラスタリング**: 画像を背景と前景に分離し、押印を抽出します
    - **色空間フィルタリング**: 特定の色（赤、青、黒など）の押印を抽出します
    - **エッジ検出**: 画像のエッジを検出し、閉じた輪郭を押印として抽出します
    """)
    
    # サイドバーでパラメータを設定
    st.sidebar.header("パラメータ設定")
    
    # 抽出方法の選択
    extraction_method = st.sidebar.selectbox(
        "抽出方法",
        ["K-meansクラスタリング", "色空間フィルタリング", "エッジ検出"]
    )
    
    # 共通パラメータ
    min_area = st.sidebar.slider("最小面積", 10, 5000, 100, 10)
    blur_size = st.sidebar.slider("ぼかしサイズ", 1, 15, 5, 2)
    if blur_size % 2 == 0:  # カーネルサイズは奇数である必要がある
        blur_size += 1
    
    # 抽出方法に応じた追加パラメータ
    if extraction_method == "色空間フィルタリング":
        color_range = st.sidebar.selectbox(
            "抽出する色",
            ["赤", "青", "黒"]
        )
        color_map = {"赤": "red", "青": "blue", "黒": "black"}
        selected_color = color_map[color_range]
    
    elif extraction_method == "エッジ検出":
        canny_low = st.sidebar.slider("Canny低閾値", 10, 200, 50, 10)
        canny_high = st.sidebar.slider("Canny高閾値", 50, 300, 150, 10)
    
    # 画像アップロード
    uploaded_file = st.file_uploader("押印がある画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 画像を読み込む
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 元の画像を表示
        st.subheader("元の画像")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="アップロードされた画像")
        
        # 押印抽出処理
        with st.spinner("押印を抽出中..."):
            if extraction_method == "K-meansクラスタリング":
                stamps, mask, stamp_coords = extract_stamps_kmeans(
                    img, 
                    min_area=min_area, 
                    blur_kernel=(blur_size, blur_size)
                )
            
            elif extraction_method == "色空間フィルタリング":
                stamps, mask, stamp_coords = extract_stamps_color(
                    img, 
                    color_range=selected_color, 
                    min_area=min_area, 
                    blur_kernel=(blur_size, blur_size)
                )
            
            elif extraction_method == "エッジ検出":
                stamps, mask, stamp_coords = extract_stamps_edge(
                    img, 
                    min_area=min_area, 
                    blur_kernel=(blur_size, blur_size),
                    canny_low=canny_low,
                    canny_high=canny_high
                )
        
        # 結果表示
        if len(stamps) > 0:
            st.success(f"{len(stamps)}個の押印が見つかりました！")
            
            # マスク画像を表示
            st.subheader("抽出マスク")
            st.image(mask, caption="押印抽出マスク")
            
            # 元画像に押印の位置を表示
            result_img = img.copy()
            for x, y, w, h in stamp_coords:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            st.subheader("検出結果")
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="検出された押印")
            
            # 抽出された押印を表示
            st.subheader("抽出された押印")
            cols = st.columns(min(len(stamps), 5))
            for i, stamp in enumerate(stamps):
                with cols[i % len(cols)]:
                    st.image(cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB), caption=f"押印 {i+1}")
                    
                    # ダウンロードボタン
                    stamp_pil = Image.fromarray(cv2.cvtColor(stamp, cv2.COLOR_BGR2RGB))
                    buf = io.BytesIO()
                    stamp_pil.save(buf, format="PNG")
                    st.download_button(
                        label=f"押印 {i+1} をダウンロード",
                        data=buf.getvalue(),
                        file_name=f"stamp_{i+1}.png",
                        mime="image/png"
                    )
        else:
            st.warning("押印が見つかりませんでした。パラメータを調整してみてください。")
            st.image(mask, caption="抽出マスク（押印なし）")

if __name__ == "__main__":
    main() 