import streamlit as st
import cv2
import numpy as np
from sklearn import cluster
import tempfile
import os
from PIL import Image

def extract_stamps(image_path, min_area=100):
    """
    画像から押印部分を抽出する関数
    
    Parameters
    ----------
    image_path : str
        入力画像のパス
    min_area : int, optional
        抽出する押印の最小面積、デフォルトは100
        
    Returns
    -------
    list
        抽出された押印画像のリスト
    """
    img = cv2.imread(image_path)
    if img is None:
        st.error("エラー: 画像を読み込めません。")
        return []
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ノイズ除去
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # K-meansクラスタリングのための準備
    pixels = blurred.ravel().reshape(-1, 1).astype(np.float32)
    
    # K-meansクラスタリング（背景と押印の2クラスタに分離）
    kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(pixels)
    labels = kmeans.predict(pixels)
    
    # 少数派のクラスタを押印と仮定
    count0, count1 = np.sum(labels == 0), np.sum(labels == 1)
    mask = (labels == 0).reshape(gray.shape) if count0 < count1 else (labels == 1).reshape(gray.shape)
    
    # 連結成分の検出
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    
    stamps = []
    for i in range(1, num_labels):  # 0はバックグラウンド
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            stamp_img = img[y:y+h, x:x+w]
            stamps.append((stamp_img, (x, y, w, h)))
    
    return stamps

def main():
    st.title("押印抽出ツール")
    st.write("画像から押印部分を抽出するツールです。")
    
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
    
    min_area = st.slider("最小押印面積", min_value=50, max_value=5000, value=500, step=50,
                        help="この値より小さい領域は押印として検出されません")
    
    if uploaded_file is not None:
        # 一時ファイルとして保存
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # 画像を表示
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_column_width=True)
        
        if st.button("押印を抽出"):
            with st.spinner("処理中..."):
                # 押印抽出処理
                stamps = extract_stamps(temp_path, min_area)
                
                if not stamps:
                    st.warning("押印が検出されませんでした。最小面積の値を調整してみてください。")
                else:
                    st.success(f"{len(stamps)}個の押印が検出されました！")
                    
                    # 元の画像を読み込み、検出された領域を表示
                    img = cv2.imread(temp_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # 検出された領域を赤枠で囲む
                    for _, (x, y, w, h) in stamps:
                        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    st.image(img_rgb, caption="検出された押印領域", use_column_width=True)
                    
                    # 個々の押印を表示
                    st.write("### 抽出された押印")
                    cols = st.columns(min(len(stamps), 3))
                    for i, (stamp_img, _) in enumerate(stamps):
                        with cols[i % 3]:
                            stamp_rgb = cv2.cvtColor(stamp_img, cv2.COLOR_BGR2RGB)
                            st.image(stamp_rgb, caption=f"押印 {i+1}", use_column_width=True)
        
        # 一時ファイルを削除
        os.unlink(temp_path)

if __name__ == "__main__":
    main()
