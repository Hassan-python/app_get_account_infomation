import os
import sys
import subprocess

def run_streamlit_app(app_file):
    """
    Streamlitアプリを実行するための関数
    
    Parameters:
    -----------
    app_file : str
        実行するStreamlitアプリのファイル名
    """
    try:
        # Pythonモジュールとして実行
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_file])
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("\n代替方法を試しています...")
        
        # 代替方法: streamlitモジュールを直接インポート
        try:
            import streamlit.web.cli as stcli
            sys.argv = ["streamlit", "run", app_file]
            stcli.main()
        except Exception as e2:
            print(f"代替方法でもエラーが発生しました: {e2}")
            print("\nStreamlitが正しくインストールされているか確認してください。")
            print("pip install streamlit --force-reinstall")

if __name__ == "__main__":
    print("押印抽出アプリを起動しています...")
    
    # 基本的なアプリを実行
    app_file = "stamp_extractor_app.py"
    
    # コマンドライン引数があれば、それを使用
    if len(sys.argv) > 1:
        app_file = sys.argv[1]
    
    # アプリを実行
    run_streamlit_app(app_file) 