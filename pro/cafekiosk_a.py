import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import cv2
from PIL import Image, ImageTk

class CafeKiosk:
    def __init__(self, root):
        self.root = root
        self.root.title("AI 카페 키오스크")
        self.root.geometry("1024x768")
        self.root.configure(bg='#FFFFFF')
        
        # 변수 초기화
        self.camera = None
        self.is_camera_running = False
        
        # GUI 생성
        self.create_gui()
        
        # 결과 체크 타이머 시작
        self.root.after(1000, self.check_results)

    def create_gui(self):
        """GUI 생성"""
        # 메인 컨테이너
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # 좌측 패널
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 우측 패널
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 좌측 패널 내용
        self.create_left_panel()
        
        # 우측 패널 내용
        self.create_right_panel()

    def create_left_panel(self):
        """좌측 패널 내용 생성"""
        # 타이틀
        title = ttk.Label(
            self.left_panel,
            text="AI 카페 키오스크",
            font=('맑은 고딕', 24, 'bold')
        )
        title.pack(pady=20)
        
        # 메뉴 목록
        self.menu_text = tk.Text(
            self.left_panel,
            height=20,
            width=40,
            font=('맑은 고딕', 12)
        )
        self.menu_text.pack(pady=10)

    def create_right_panel(self):
        """우측 패널 내용 생성"""
        # 분석 결과
        self.result_label = ttk.Label(
            self.right_panel,
            text="분석 결과",
            font=('맑은 고딕', 20)
        )
        self.result_label.pack(pady=20)
        
        # 추천 메뉴
        self.recommendation_text = tk.Text(
            self.right_panel,
            height=15,
            width=40,
            font=('맑은 고딕', 12)
        )
        self.recommendation_text.pack(pady=10)

    def check_results(self):
        """결과 파일 체크 및 업데이트"""
        try:
            if os.path.exists("analysis_results.json"):
                with open("analysis_results.json", "r", encoding='utf-8') as f:
                    data = json.load(f)
                
                # 결과 표시
                self.update_results(data)
        except Exception as e:
            print(f"결과 체크 오류: {e}")
        
        # 1초마다 체크
        self.root.after(1000, self.check_results)

    def update_results(self, data):
        """결과 업데이트"""
        # 메뉴 추천 텍스트 업데이트
        self.recommendation_text.delete(1.0, tk.END)
        text = f"""
연령대: {data['연령대']}
나이: {data['나이']}세

=== 추천 메뉴 ===
"""
        for i, menu in enumerate(data['추천메뉴'], 1):
            text += f"{i}. {menu}\n"
        
        self.recommendation_text.insert(tk.END, text)