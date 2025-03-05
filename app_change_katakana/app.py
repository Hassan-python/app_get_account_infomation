import streamlit as st
import easyocr
from PIL import Image
import io
import pandas as pd
import numpy as np
import cv2
import zipfile
from datetime import datetime

def preprocess_image(image):
    # PIL ImageをOpenCV形式に変換
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 二値化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ノイズ除去
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # コントラスト強調
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def get_bbox_area(bbox):
    # バウンディングボックスの面積を計算
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width * height

def merge_overlapping_boxes(boxes, overlap_threshold=0.3):
    if not boxes:
        return []
    
    # バウンディングボックスを面積でソート（大きい順）
    boxes = sorted(boxes, key=get_bbox_area, reverse=True)
    merged_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        current_area = get_bbox_area(current_box)
        
        # 他のボックスとの重なりをチェック
        i = 0
        while i < len(boxes):
            other_box = boxes[i]
            other_area = get_bbox_area(other_box)
            
            # 重なり面積を計算
            x1 = max(min(point[0] for point in current_box), min(point[0] for point in other_box))
            y1 = max(min(point[1] for point in current_box), min(point[1] for point in other_box))
            x2 = min(max(point[0] for point in current_box), max(point[0] for point in other_box))
            y2 = min(max(point[1] for point in current_box), max(point[1] for point in other_box))
            
            if x2 > x1 and y2 > y1:
                overlap_area = (x2 - x1) * (y2 - y1)
                min_area = min(current_area, other_area)
                
                if overlap_area / min_area > overlap_threshold:
                    # 重なっているボックスをマージ
                    boxes.pop(i)
                    continue
            
            i += 1
        
        merged_boxes.append(current_box)
    
    return merged_boxes

def extract_text_regions(image, result, min_area=100, max_area=None, overlap_threshold=0.3):
    # テキストを含む領域を特定
    text_boxes = []
    for detection in result:
        text = detection[1]
        if text.strip():  # 空でないテキストの場合
            text_boxes.append(detection[0])
    
    # 重なっている領域をマージ
    merged_boxes = merge_overlapping_boxes(text_boxes, overlap_threshold)
    
    # 各領域を切り出し
    regions = []
    for box in merged_boxes:
        # バウンディングボックスの座標を整数に変換
        x_coords = [int(point[0]) for point in box]
        y_coords = [int(point[1]) for point in box]
        
        # 画像の範囲内に収める
        x_min = max(0, min(x_coords))
        x_max = min(image.shape[1], max(x_coords))
        y_min = max(0, min(y_coords))
        y_max = min(image.shape[0], max(y_coords))
        
        # 領域の面積を計算
        area = (x_max - x_min) * (y_max - y_min)
        
        # 面積条件をチェック
        if area >= min_area and (max_area is None or area <= max_area):
            # 領域を切り出し
            region = image[y_min:y_max, x_min:x_max]
            regions.append((region, (x_min, y_min, x_max, y_max)))
    
    return regions

def create_zip_file(regions):
    # ZIPファイルを作成
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, (region, _) in enumerate(regions, 1):
            # OpenCV形式からPIL形式に変換
            region_pil = Image.fromarray(region)
            # 画像をバイトストリームに保存
            img_buffer = io.BytesIO()
            region_pil.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            # ZIPファイルに追加
            zip_file.writestr(f'region_{i}.png', img_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer

def main():
    st.title("テキスト領域検出アプリ")
    st.write("画像からテキストを含む領域を検出し、切り出します。")

    # パラメータ設定
    st.sidebar.header("検出パラメータ")
    min_area = st.sidebar.number_input("最小領域面積", min_value=0, value=100, step=10)
    max_area = st.sidebar.number_input("最大領域面積", min_value=0, value=10000, step=100)
    overlap_threshold = st.sidebar.slider("重なり判定閾値", min_value=0.0, max_value=1.0, value=0.3, step=0.1)

    uploaded_file = st.file_uploader("画像ファイルを選択", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 画像を表示
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_container_width=True)

        # OCRの実行
        with st.spinner("テキスト領域を検出中..."):
            # 画像の前処理
            processed_image = preprocess_image(image)
            
            reader = easyocr.Reader(['ja'])
            result = reader.readtext(processed_image)
            
            # テキスト領域の検出と切り出し
            regions = extract_text_regions(processed_image, result, min_area, max_area, overlap_threshold)
            
            if regions:
                st.subheader("検出されたテキスト領域")
                
                # 検出された領域を表示
                for i, (region, coords) in enumerate(regions, 1):
                    st.write(f"領域 {i}")
                    # OpenCV形式からPIL形式に変換
                    region_pil = Image.fromarray(region)
                    st.image(region_pil, caption=f"領域 {i} (座標: {coords})", use_container_width=True)
                    
                    # 個別ダウンロードボタン
                    img_buffer = io.BytesIO()
                    region_pil.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    st.download_button(
                        label=f"領域 {i} をダウンロード",
                        data=img_buffer,
                        file_name=f'region_{i}.png',
                        mime='image/png'
                    )
                
                # まとめてダウンロードボタン
                zip_buffer = create_zip_file(regions)
                st.download_button(
                    label="すべての領域をZIPでダウンロード",
                    data=zip_buffer,
                    file_name=f'text_regions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                    mime='application/zip'
                )
                
                # 検出された領域の情報を表示
                st.subheader("検出された領域の情報")
                region_info = []
                for i, (region, coords) in enumerate(regions, 1):
                    x_min, y_min, x_max, y_max = coords
                    region_info.append({
                        "領域番号": i,
                        "X座標": f"{x_min}-{x_max}",
                        "Y座標": f"{y_min}-{y_max}",
                        "幅": x_max - x_min,
                        "高さ": y_max - y_min,
                        "面積": (x_max - x_min) * (y_max - y_min)
                    })
                df_info = pd.DataFrame(region_info)
                st.dataframe(df_info, use_container_width=True)
            else:
                st.warning("テキストを含む領域は検出されませんでした。パラメータを調整してみてください。")

if __name__ == "__main__":
    main() 