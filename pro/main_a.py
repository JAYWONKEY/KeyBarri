import tkinter as tk
import threading
import argparse
from face_analyzer_a import FaceAnalyzer
from cafekiosk_a import CafeKiosk

def start_kiosk():
    try:
        print("=== AI 카페 키오스크 UI 시작 ===")
        root = tk.Tk()
        app = CafeKiosk(root)
        root.mainloop()
    except Exception as e:
        print(f"키오스크 UI 오류: {str(e)}")

def start_face_analyzer():
    try:
        print("=== 얼굴 인식 및 분석 시스템 시작 ===")
        analyzer = FaceAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"얼굴 인식 오류: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='AI 카페 키오스크 시스템')
    parser.add_argument('--mode', type=str, default='full', 
                      choices=['kiosk', 'analyzer', 'full'],
                      help='실행 모드 선택')
    args = parser.parse_args()
    
    if args.mode == 'full':
        analyzer_thread = threading.Thread(target=start_face_analyzer)
        analyzer_thread.daemon = True
        analyzer_thread.start()
        start_kiosk()
    elif args.mode == 'kiosk':
        start_kiosk()
    elif args.mode == 'analyzer':
        start_face_analyzer()

if __name__ == "__main__":
    print("\n===== AI 카페 키오스크 시스템 =====")
    main()