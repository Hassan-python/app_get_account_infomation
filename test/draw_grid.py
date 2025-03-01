from PIL import Image, ImageDraw
import argparse
import os

def draw_grid(image_path, output_path=None, grid_size=50, line_color=(255, 0, 0), line_width=1):
    """
    画像にグリッドを描画する関数
    
    Parameters:
    -----------
    image_path : str
        入力画像のパス
    output_path : str, optional
        出力画像のパス。指定がない場合は元のファイル名に '_grid' を追加
    grid_size : int, optional
        グリッドのセルサイズ（ピクセル単位）、デフォルトは50
    line_color : tuple, optional
        グリッド線の色（R, G, B）、デフォルトは赤色
    line_width : int, optional
        グリッド線の太さ、デフォルトは1
    """
    try:
        # 画像を開く
        img = Image.open(image_path)
        
        # 描画オブジェクトを作成
        draw = ImageDraw.Draw(img)
        
        # 画像のサイズを取得
        width, height = img.size
        
        # 横線を描画
        for y in range(0, height, grid_size):
            draw.line([(0, y), (width, y)], fill=line_color, width=line_width)
        
        # 縦線を描画
        for x in range(0, width, grid_size):
            draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)
        
        # 出力パスが指定されていない場合、デフォルトのパスを生成
        if output_path is None:
            file_name, file_ext = os.path.splitext(image_path)
            output_path = f"{file_name}_grid{file_ext}"
        
        # 画像を保存
        img.save(output_path)
        print(f"グリッド付き画像を保存しました: {output_path}")
        return True
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

def main():
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='画像にグリッドを描画するツール')
    parser.add_argument('image_path', help='入力画像のパス')
    parser.add_argument('-o', '--output', help='出力画像のパス')
    parser.add_argument('-s', '--size', type=int, default=50, help='グリッドのセルサイズ（ピクセル単位）')
    parser.add_argument('-c', '--color', nargs=3, type=int, default=[255, 0, 0], help='グリッド線の色（R G B）')
    parser.add_argument('-w', '--width', type=int, default=1, help='グリッド線の太さ')
    
    args = parser.parse_args()
    
    # 関数を呼び出し
    draw_grid(
        args.image_path, 
        args.output, 
        args.size, 
        tuple(args.color), 
        args.width
    )

if __name__ == "__main__":
    main() 