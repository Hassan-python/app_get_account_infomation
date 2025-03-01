import streamlit as st
import cv2
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from PIL import Image
import io

st.set_page_config(page_title="押印抽出アプリ", layout="wide")

def extract_stamps(img, min_area=100, blur_kernel=(5, 5)):
    """
    画像から押印部分を抽出する関数
    
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
    
    return stamps, mask.astype(np.uint8) * 255, stamp_coords

def main():
    st.title("押印抽出アプリ")
    
    st.markdown("""
    このアプリは画像から押印部分を自動的に抽出します。
    
    ### 使い方
    1. 左側のサイドバーでパラメータを調整します
    2. 画像をアップロードします
    3. 抽出結果を確認します
    
    ### 技術的な詳細
    - グレースケール変換とノイズ除去
    - K-meansクラスタリングによる背景と押印の分離
    - 連結成分分析による押印領域の特定
    """)
    
    # サイドバーでパラメータを設定
    st.sidebar.header("パラメータ設定")
    min_area = st.sidebar.slider("最小面積", 10, 5000, 100, 10)
    blur_size = st.sidebar.slider("ぼかしサイズ", 1, 15, 5, 2)
    if blur_size % 2 == 0:  # カーネルサイズは奇数である必要がある
        blur_size += 1
    
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
            stamps, mask, stamp_coords = extract_stamps(img, min_area=min_area, blur_kernel=(blur_size, blur_size))
        
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