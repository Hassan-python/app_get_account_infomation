import os
import sys
import subprocess
import site

def check_streamlit_installation():
    """
    Streamlitのインストール状況とPATH環境変数を確認する関数
    """
    print("=== システム情報 ===")
    print(f"Python実行パス: {sys.executable}")
    print(f"Pythonバージョン: {sys.version}")
    
    # PATH環境変数の確認
    print("\n=== PATH環境変数 ===")
    path_entries = os.environ.get('PATH', '').split(os.pathsep)
    for i, path in enumerate(path_entries):
        print(f"{i+1}. {path}")
    
    # サイトパッケージの確認
    print("\n=== サイトパッケージのパス ===")
    for path in site.getsitepackages():
        print(path)
    
    # Streamlitのインストール確認
    print("\n=== Streamlitのインストール状況 ===")
    try:
        # pipでインストールされたパッケージの確認
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                               capture_output=True, text=True)
        packages = result.stdout.splitlines()
        
        streamlit_found = False
        for package in packages:
            if package.lower().startswith("streamlit "):
                streamlit_found = True
                print(f"インストール済み: {package}")
                break
        
        if not streamlit_found:
            print("Streamlitはインストールされていません。")
        
        # Streamlitの実行ファイルを探す
        print("\n=== Streamlit実行ファイルの検索 ===")
        for path in path_entries:
            streamlit_path = os.path.join(path, "streamlit")
            streamlit_path_exe = os.path.join(path, "streamlit.exe")
            
            if os.path.exists(streamlit_path):
                print(f"見つかりました: {streamlit_path}")
            
            if os.path.exists(streamlit_path_exe):
                print(f"見つかりました: {streamlit_path_exe}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    check_streamlit_installation() 