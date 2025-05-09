import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QGridLayout, 
                            QFrame, QDialog, QComboBox, QCheckBox, QListWidget, 
                            QListWidgetItem, QScrollArea, QTextEdit, QLineEdit,
                            QMessageBox, QSpacerItem)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon
import re # 25.05.01 추가
import pandas
from pro_deep import get_age_from_camera,FaceRecognitionKiosk
from stt_whisper import run_stt_pipeline

# preprocessing.py에서 필요한 함수 import
try:
    from main import (
        load_csv_data,
        text_to_speech,
        filter_menu_by_health,
        recommend_by_age,
        recommend_by_age_and_disease,
        recommend_menu_only,
        rag_pipeline
    )
    import pandas as pd
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"모듈 로딩 오류: {e}")
    print("필요한 패키지를 설치하세요:")
    print("pip install PyQt5 pandas numpy faiss-cpu sentence-transformers pygame gtts")
    sys.exit(1)

# 메인 윈도우 클래스
class MegaKioskUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 기본 설정
        self.setWindowTitle("메가 커피 키오스크")
        self.setGeometry(100, 100, 800, 1000)
        
        # 메뉴 데이터 로드
        self.load_menu_data()
        
        # RAG 모델 초기화
        self.initialize_rag()
        
        # 카트(장바구니) 초기화
        self.cart = {}  # {menu_name: {'price': price, 'count': count}}
        
        self.conversation_history = {}  # ✅ 대화 이력 초기화 5.7추가 

        # UI 설정
        self.init_ui()


            #5.7 추가 STT버튼
        # ✅ 여기서 mic_button을 생성하고 cart_layout에 추가!
        mic_button = QPushButton("🎤")
        mic_button.setFixedSize(50, 50) #60번째 줄 + 61번주 마이크 생성 코드
        #스타일 지정
        mic_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 20px;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        mic_button.clicked.connect(self.run_voice_rag)#5.7 추가 버튼 클릭 이벤트 연결

    def run_voice_rag(self):  # 👈 꼭 있어야 합니다!5.7
        try:
            from stt_whisper import run_stt_pipeline
            question, answer = run_stt_pipeline(
                self.menu_texts,
                self.embedder,
                self.index,
                self.menu_df,
                self.conversation_history
            )
            QMessageBox.information(self, "RAG 응답", f"질문: {question}\n\n답변: {answer}")
        except Exception as e:
            QMessageBox.warning(self, "실패", str(e))
    
    def load_menu_data(self):
        """메뉴 데이터 로드"""
        try:
            # CSV 파일 경로 - 실제 프로젝트 구조에 맞게 조정
            csv_path = "123\por0502\process_data.csv"
            
            # 원본 데이터 로드
            #self.menu_texts = pd.load_csv_data(csv_path, encoding='utf-8')
            
            # 메뉴 데이터 구조화
            self.menu_df = pd.read_csv(csv_path, encoding='utf-8')
            
            print(f"메뉴 데이터 로드 완료: {len(self.menu_df)}개 메뉴")
            # 25.05.01 추가 메뉴 텍스트 생성에 분류 정보 추가
            # self.menu_texts, self.embedder, self.index = self.initialize_rag(self.menu_df)

            # 필요한 열만 선택
            # self.menu_df = self.menu_df[['가격', '이름', '칼로리(kcal)', '탄수화물(g)', '당류(g)', 
            #                             '단백질(g)', '지방(g)', '포화지방(g)', '트랜스지방(g)', 
            #                             '나트륨(mg)', '콜레스테롤(mg)', '카페인(mg)', '알레르기 유발물질', '연령']]
            
        except Exception as e:
            print(f"메뉴 데이터 로드 오류: {str(e)}")
            # 기본 메뉴 데이터 생성
            self.create_sample_menu_data()
    
    def create_sample_menu_data(self):
        """샘플 메뉴 데이터 생성 (CSV 로드 실패 시)"""
        print("샘플 메뉴 데이터 생성")
        
        # 기본 메뉴 정보
        self.menu_df = pd.DataFrame({
            '카테고리번호': list(range(1, 16)),
            'HOT/ICE': ['HOT', 'ICE'] * 7 + ['HOT'],
            '분류': ['커피(HOT)'] * 5 + ['커피(ICE)'] * 5 + ['차/티'] * 5,
            '가격': [4000, 4500, 5000, 5500, 4000, 4500, 5000, 5500, 6000, 6500, 4500, 4000, 4500, 5000, 5500],
            '이름': [
                '아메리카노(HOT)', '카페라떼(HOT)', '바닐라라떼(HOT)', '카라멜마키아또(HOT)', '에스프레소(HOT)',
                '아메리카노(ICE)', '카페라떼(ICE)', '바닐라라떼(ICE)', '카라멜마키아또(ICE)', '아이스티(ICE)',
                '녹차라떼', '홍차', '유자차', '페퍼민트', '캐모마일'
            ],
            '칼로리(kcal)': np.random.randint(10, 500, 15),
            '탄수화물(g)': np.random.uniform(0, 50, 15),
            '당류(g)': np.random.uniform(0, 20, 15),
            '단백질(g)': np.random.uniform(0, 10, 15),
            '지방(g)': np.random.uniform(0, 10, 15),
            '포화지방(g)': np.random.uniform(0, 5, 15),
            '트랜스지방(g)': np.random.uniform(0, 1, 15),
            '나트륨(mg)': np.random.uniform(0, 200, 15),
            '콜레스테롤(mg)': np.random.uniform(0, 50, 15),
            '카페인(mg)': np.random.uniform(0, 250, 15),
            '알레르기 유발물질': ['-'] * 15,
            # 25.05.01 수정 : 성인 (청년, 중년, 장년) 묶음 
            '연령': ['어린이, 청소년, 성인, 노인'] * 15
        })
        
    def initialize_rag(self):
        """RAG 모델 초기화"""
        try:
            # 메뉴 텍스트 생성
            if hasattr(self, 'menu_df') and not self.menu_df.empty:
                # 분류 컬럼이 있는지 확인
                if '분류' in self.menu_df.columns:
                    menu_texts = self.menu_df.apply(lambda row: f"{row['이름']} {row['가격']}원. 분류: {row['분류']}.", axis=1).tolist()
                else:
                    menu_texts = self.menu_df.apply(lambda row: f"{row['이름']} {row['가격']}원.", axis=1).tolist()
                
                # 카테고리별 요약 정보 추가
                summary_texts = []
                if '분류' in self.menu_df.columns:
                    categories = self.menu_df['분류'].unique()
                    for category in categories:
                        items = self.menu_df[self.menu_df['분류'] == category]['이름'].tolist()
                        count = len(items)
                        summary_texts.append(f"{category} 메뉴는 총 {count}개로 {', '.join(items[:5])} 등이 있습니다.")
                
                # 요약 텍스트를 메뉴 텍스트에 추가
                all_texts = summary_texts + menu_texts
                
                # 임베딩 모델 초기화
                print("RAG 모델 초기화 중...")
                self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                # 메뉴 텍스트 임베딩
                embeddings = self.embedder.encode(all_texts)
                
                # FAISS 인덱스 구축
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings, dtype=np.float32))
                
                # 전체 텍스트 저장
                self.menu_texts = all_texts
                
                print("RAG 모델 초기화 완료")
            else:
                print("메뉴 데이터가 로드되지 않아 RAG 모델을 초기화할 수 없습니다")
                self.menu_texts = []
                self.embedder = None
                self.index = None
                
        except Exception as e:
            print(f"RAG 모델 초기화 오류: {str(e)}")
            self.menu_texts = []
            self.embedder = None
            self.index = None
    
                        
        
    def init_ui(self):
        """메인 UI 초기화"""
        # 중앙 위젯 및 레이아웃
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 타이틀 바
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #FFCC00; border: none;")
        title_frame.setFixedHeight(70)
        
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("MEGA COFFEE")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title_label)
        
        # 카테고리 바
        category_frame = QFrame()
        category_frame.setStyleSheet("background-color: #FFCC00; border: none;")
        
        category_layout = QGridLayout(category_frame)
        category_layout.setSpacing(2)
        
        # 카테고리 버튼 생성
        categories = [
            "디카페인", "추천메뉴", "커피(ICE)", "커피(HOT)",
            "오늘의메뉴", "스무디", "티/차", "AI 서비스 이용하기"
        ]
        
        self.category_buttons = []
        for i, category in enumerate(categories):
            btn = QPushButton(category)
            btn.setFixedHeight(60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #333;
                    color: white;
                    border-radius: 10px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #E54F40;
                }
            """)
            
            # AI 서비스 버튼 강조
            if category == "AI 서비스 이용하기":
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #E54F40;
                        color: white;
                        border-radius: 10px;
                        font-weight: bold;
                        font-size: 14px;
                    }
                    QPushButton:hover {
                        background-color: #D44539;
                    }
                """)
                btn.clicked.connect(self.show_ai_service_dialog)
            
            row, col = i // 4, i % 4
            category_layout.addWidget(btn, row, col)
            self.category_buttons.append(btn)
        
        # 메뉴 그리드 영역
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #FFCC00;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        self.menu_widget = QWidget()
        self.menu_layout = QGridLayout(self.menu_widget)
        self.menu_layout.setSpacing(10)
        
        scroll_area.setWidget(self.menu_widget)
        
        # 장바구니 영역
        cart_frame = QFrame()
        cart_frame.setFixedHeight(70)
        cart_frame.setStyleSheet("background-color: #f5f5f5; border-top: 1px solid #ddd;")
        
        cart_layout = QHBoxLayout(cart_frame)
        
        self.cart_label = QLabel("장바구니 (0개)")
        self.cart_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        self.payment_button = QPushButton("0원 결제하기")
        self.payment_button.setStyleSheet("""
            QPushButton {
                background-color: #E54F40;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D44539;
            }
        """)


        self.payment_button.clicked.connect(self.process_payment)
        
        cart_layout.addWidget(self.cart_label)
        cart_layout.addWidget(self.payment_button)
        #5.7추가
        mic_button = QPushButton("🎤")
        mic_button.setFixedSize(50, 50)
        mic_button.setStyleSheet("""
    QPushButton {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        border-radius: 25px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
""")
        mic_button.clicked.connect(self.run_voice_rag)
        cart_layout.addWidget(mic_button)

        # 메인 레이아웃에 위젯 추가
        main_layout.addWidget(title_frame)
        main_layout.addWidget(category_frame)
        main_layout.addWidget(scroll_area, 1)
        main_layout.addWidget(cart_frame)
        
        self.setCentralWidget(central_widget)
        
        # 메뉴 표시
        self.display_menu()
    
    def display_menu(self):
        """메뉴 표시"""
        # 기존 메뉴 위젯 제거
        while self.menu_layout.count():
            item = self.menu_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # 새 메뉴 위젯 생성
        menu_count = min(12, len(self.menu_df))  # 최대 12개 메뉴 표시
        
        for i in range(menu_count):
            menu_data = self.menu_df.iloc[i]
            
            # 메뉴 프레임
            menu_frame = QFrame()
            menu_frame.setFixedSize(180, 200)
            menu_frame.setStyleSheet("""
                QFrame {
                    background-color: white;
                    border-radius: 10px;
                    border: 1px solid #ddd;
                }
                QFrame:hover {
                    border: 2px solid #E54F40;
                }
            """)
            
            # 메뉴 레이아웃
            menu_layout = QVBoxLayout(menu_frame)
            
            # 메뉴 이미지 (임시)
            image_label = QLabel()
            image_label.setFixedSize(120, 100)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet(f"background-color: #f0f0f0; border-radius: 5px;")
            image_label.setText(menu_data['이름'][0])  # 메뉴 이름의 첫 글자
            
            # 메뉴 이름
            name_label = QLabel(menu_data['이름'])
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            # 메뉴 가격
            price_label = QLabel(f"{menu_data['가격']}원")
            price_label.setAlignment(Qt.AlignCenter)
            price_label.setStyleSheet("color: #E54F40; font-size: 14px;")
            
            # 레이아웃에 위젯 추가
            menu_layout.addWidget(image_label)
            menu_layout.addWidget(name_label)
            menu_layout.addWidget(price_label)
            
            # 클릭 이벤트 연결
            menu_frame.mousePressEvent = lambda e, name=menu_data['이름'], price=menu_data['가격']: self.add_to_cart(name, price)
            
            # 그리드에 메뉴 추가
            row, col = i // 3, i % 3
            self.menu_layout.addWidget(menu_frame, row, col)
    
    def add_to_cart(self, name, price):
        """장바구니에 메뉴 추가"""
        if name in self.cart:
            self.cart[name]['count'] += 1
        else:
            self.cart[name] = {'price': price, 'count': 1}
        
        # 장바구니 UI 업데이트
        self.update_cart_ui()
        
        # 메시지 표시
        QMessageBox.information(self, "메뉴 추가", f"{name}이(가) 장바구니에 추가되었습니다.")
    
    def update_cart_ui(self):
        """장바구니 UI 업데이트"""
        # 총 아이템 개수
        total_count = sum(item['count'] for item in self.cart.values())
        
        # 총 가격
        total_price = sum(item['price'] * item['count'] for item in self.cart.values())
        
        # UI 업데이트
        self.cart_label.setText(f"장바구니 ({total_count}개)")
        self.payment_button.setText(f"{total_price}원 결제하기")
    
    def process_payment(self):
        """결제 처리"""
        if not self.cart:
            QMessageBox.warning(self, "결제 오류", "장바구니가 비어 있습니다.")
            return
        
        # 장바구니 내용 문자열 생성
        cart_text = "\n".join([f"{name} x {item['count']} = {item['price'] * item['count']}원" 
                              for name, item in self.cart.items()])
        
        # 총 가격
        total_price = sum(item['price'] * item['count'] for item in self.cart.values())
        
        # 확인 메시지
        reply = QMessageBox.question(self, "결제 확인", 
                                    f"다음 메뉴를 결제하시겠습니까?\n\n{cart_text}\n\n총액: {total_price}원",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            QMessageBox.information(self, "결제 완료", "결제가 완료되었습니다. 이용해 주셔서 감사합니다!")
            
            # 장바구니 초기화
            self.cart = {}
            self.update_cart_ui()
    
    def show_ai_service_dialog(self):
        """AI 서비스 다이얼로그 표시"""
        dialog = AIServiceDialog(self, self.menu_df)
        dialog.exec_()
        
        # 장바구니 UI 업데이트 (추천 메뉴가 추가됐을 수 있음)
        self.update_cart_ui()

# AI 서비스 다이얼로그 클래스
class AIServiceDialog(QDialog):
    def __init__(self, parent, menu_df):
        super().__init__(parent)
        self.parent = parent
        self.menu_df = menu_df
        
         # 부모로부터 필요한 데이터 가져오기
        self.menu_texts = parent.menu_texts
        self.embedder = parent.embedder
        self.index = parent.index

        # 기본 설정
        self.setWindowTitle("AI 메뉴 추천 서비스")
        self.setFixedSize(600, 700)
        self.setStyleSheet("background-color: white;")
        
        # 초기 UI 설정 (메인 화면)
        self.init_ui()
    
    def init_ui(self):
        """다이얼로그 UI 초기화"""
        self.main_layout = QVBoxLayout(self)
        
        # 타이틀 바
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #FFCC00;")
        title_frame.setFixedHeight(60)
        
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("AI 메뉴 추천 서비스")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        
        close_button = QPushButton("X")
        close_button.setFixedSize(30, 30)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: #E54F40;
                color: white;
                border-radius: 15px;
                font-weight: bold;
            }
        """)
        close_button.clicked.connect(self.reject)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(close_button)
        
        # 본문 위젯 및 레이아웃
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignCenter)
        self.content_layout.setSpacing(20)
        
        # 환영 메시지
        welcome_label = QLabel("어서오세요! 건강과 취향에 맞는 메뉴를\n추천해 드립니다")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 16px; margin: 20px 0;")
        
        # 서비스 버튼들
        services = [
            ("얼굴 인식으로 맞춤 추천 받기", self.show_face_recognition),
            ("빠른 추천 받기", self.show_quick_recommendation),
            ("건강 맞춤 추천 받기", self.show_health_recommendation),
            ("메뉴에 대해 물어보기", self.show_chat_interface),
            ("메인 화면으로 돌아가기", self.reject)
        ]
        
        for text, func in services:
            btn = QPushButton(text)
            btn.setFixedHeight(60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    font-size: 16px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #FFCC00;
                }
            """)
            
            # 얼굴 인식과 메뉴 질문 버튼 강조
            if "얼굴 인식" in text:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #E54F40;
                        color: white;
                        border-radius: 10px;
                        font-size: 16px;
                        padding: 10px;
                    }
                    QPushButton:hover {
                        background-color: #D44539;
                    }
                """)
            elif "물어보기" in text:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 10px;
                        font-size: 16px;
                        padding: 10px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
            
            btn.clicked.connect(func)
            self.content_layout.addWidget(btn)
        
        # 스크롤 영역
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none;")
        scroll_area.setWidget(self.content_widget)
        
        # 메인 레이아웃에 위젯 추가
        self.main_layout.addWidget(title_frame)
        self.main_layout.addWidget(scroll_area)
    
    def clear_content(self):
        """컨텐츠 영역 초기화"""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def add_back_button(self):
        """뒤로 가기 버튼 추가"""
        back_btn = QPushButton("처음으로 돌아가기")
        back_btn.setFixedHeight(50)
        back_btn.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border-radius: 10px;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        back_btn.clicked.connect(self.init_ui)
        self.content_layout.addWidget(back_btn)
    
    def show_face_recognition(self):
        self.clear_content()

        age = get_age_from_camera()
        if age is None:
            QMessageBox.warning(self, "인식 실패", "얼굴 인식에 실패했습니다.")
            return

        # 나이에 따라 연령대 추정
        if age < 13:
            age_group = "어린이"
        elif age < 19:
            age_group = "청소년"
        elif age < 40:
            age_group = "청년"
        elif age < 60:
            age_group = "중년"
        else:
            age_group = "노인"

        recommendation_text = recommend_by_age(self.menu_df, age_group)

        result_label = QLabel(f"추정 나이: {age}세 → {age_group}\n{recommendation_text}")
        result_label.setAlignment(Qt.AlignCenter)
        result_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px 0;")

        tts_btn = QPushButton("음성으로 들려주기")
        tts_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        tts_btn.clicked.connect(lambda: self.play_tts(result_label.text()))

        self.content_layout.addWidget(result_label)
        self.content_layout.addWidget(tts_btn)
        self.add_back_button()
        
        # 추천 메뉴 리스트
        recommended_menu_frame = QFrame()
        recommended_menu_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        recommended_menu_layout = QVBoxLayout(recommended_menu_frame)
        
        # 추천 메뉴
        recommended_menus = [
            ("아메리카노", 4000),
            ("카페라떼", 4500),
            ("바닐라라떼", 5000),
            ("에스프레소", 3500),
            ("아이스티", 4000)
        ]
        
        self.recommendation_checkboxes = []
        
        for name, price in recommended_menus:
            menu_item = QCheckBox(f"{name} ({price}원)")
            menu_item.setStyleSheet("font-size: 14px; margin: 5px;")
            recommended_menu_layout.addWidget(menu_item)
            self.recommendation_checkboxes.append((menu_item, name, price))
        
        # 장바구니 추가 버튼
        add_to_cart_btn = QPushButton("선택한 메뉴 장바구니에 추가")
        add_to_cart_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFCC00;
                border-radius: 10px;
                font-size: 16px;
                                      padding: 10px;
            }
            QPushButton:hover {
                background-color: #E5B800;
            }
        """)
        add_to_cart_btn.clicked.connect(self.add_selected_to_cart)
        
        # 컨텐츠 영역에 위젯 추가
        self.content_layout.addWidget(result_label)
        
        self.content_layout.addWidget(recommended_menu_frame)
        self.content_layout.addWidget(add_to_cart_btn)
        
        # 뒤로가기 버튼 추가
        self.add_back_button()
    # 25.05.01 2. AIServiceDialog 클래스 - 연령대 선택 부분 수정
    def show_quick_recommendation(self):
        """빠른 추천 화면 표시"""
        self.clear_content()
        
        # 연령대 선택 레이블
        age_label = QLabel("연령대를 선택하세요:")
        age_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        
        # 연령대 버튼들
        age_frame = QFrame()
        age_layout = QHBoxLayout(age_frame)

        # 25.05.01 추가 연령대 목록 수정 - "성인" 추가
        age_groups = ["어린이", "청소년", "성인", "노인"]
        
        for age in age_groups:
            btn = QPushButton(age)
            btn.setFixedHeight(50)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    font-size: 14px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #FFCC00;
                }
            """)
            btn.clicked.connect(lambda _, age=age: self.show_quick_result(age))
            age_layout.addWidget(btn)
        
        # 컨텐츠 영역에 위젯 추가
        self.content_layout.addWidget(age_label)
        self.content_layout.addWidget(age_frame)
        
        # 뒤로가기 버튼 추가
        self.add_back_button()
    
    def show_quick_result(self, age_group):
        """빠른 추천 결과 표시"""
        try:
            # preprocessing.py의 recommend_by_age 함수 호출
            recommendation_text = recommend_by_age(self.menu_df, age_group)
            
            # 결과 프레임
            result_frame = QFrame()
            result_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
            
            result_layout = QVBoxLayout(result_frame)
            
            # 결과 텍스트
            result_label = QLabel(recommendation_text)
            result_label.setWordWrap(True)
            result_label.setStyleSheet("font-size: 16px;")
            
            # 음성으로 들려주기 버튼
            tts_btn = QPushButton("음성으로 들려주기")
            tts_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            tts_btn.clicked.connect(lambda: self.play_tts(result_label.text()))
            
            result_layout.addWidget(result_label)
            result_layout.addWidget(tts_btn)
            
            # 결과 프레임을 컨텐츠 레이아웃에 추가
            # 먼저 기존 결과 프레임 제거 (있으면)
            for i in range(self.content_layout.count()):
                widget = self.content_layout.itemAt(i).widget()
                if isinstance(widget, QFrame) and widget != self.content_layout.itemAt(0).widget():
                    widget.deleteLater()
            
            # 새 결과 프레임 추가
            self.content_layout.insertWidget(2, result_frame)
            
            # 추천된 메뉴 추출
            
            menu_names = []
            pattern = r"추천되는 메뉴입니다: (.*?)\." ####1
            match = re.search(pattern, recommendation_text)

            # if "추천되는 메뉴입니다" in recommendation_text:
            #     menu_text = recommendation_text.split("추천되는 메뉴입니다: ")[1].split(".")[0]
            #     menu_names = [name.strip() for name in menu_text.split(", ")]
            
            # 추천 메뉴 리스트 생성
            if menu_names:
                # 기존 리스트 위젯 제거 (있으면)
                for i in range(self.content_layout.count()):
                    widget = self.content_layout.itemAt(i).widget()
                    if isinstance(widget, QListWidget):
                        widget.deleteLater()
                
                # 새 리스트 위젯 생성
                menu_list = QListWidget()
                menu_list.setStyleSheet("""
                    QListWidget {
                        background-color: #f5f5f5;
                        border-radius: 10px;
                        padding: 5px;
                    }
                    QListWidget::item {
                        padding: 5px;
                    }
                """)
                
                # 메뉴 아이템 추가
                self.recommendation_checkboxes = []
                for name in menu_names:
                    # 메뉴 가격 찾기
                    price = 0
                    for _, row in self.menu_df.iterrows():
                        if row['이름'] == name:
                            price = row['가격']
                            break
                    
                    item = QListWidgetItem()
                    item_widget = QWidget()
                    item_layout = QHBoxLayout(item_widget)
                    
                    checkbox = QCheckBox(f"{name} ({price}원)")
                    checkbox.setStyleSheet("font-size: 14px;")
                    
                    item_layout.addWidget(checkbox)
                    
                    item.setSizeHint(item_widget.sizeHint())
                    menu_list.addItem(item)
                    menu_list.setItemWidget(item, item_widget)
                    
                    self.recommendation_checkboxes.append((checkbox, name, price))
                
                # 메뉴 리스트 추가
                self.content_layout.insertWidget(3, menu_list)
                
                # 장바구니 추가 버튼
                add_to_cart_btn = QPushButton("선택한 메뉴 장바구니에 추가")
                add_to_cart_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FFCC00;
                        border-radius: 10px;
                        font-size: 16px;
                        padding: 10px;
                    }
                    QPushButton:hover {
                        background-color: #E5B800;
                    }
                """)
                add_to_cart_btn.clicked.connect(self.add_selected_to_cart)
                
                # 기존 장바구니 버튼 제거 (있으면)
                for i in range(self.content_layout.count()):
                    widget = self.content_layout.itemAt(i).widget()
                    if isinstance(widget, QPushButton) and widget.text() == "선택한 메뉴 장바구니에 추가":
                        widget.deleteLater()
                
                # 새 장바구니 버튼 추가
                self.content_layout.insertWidget(4, add_to_cart_btn)
        
        except Exception as e:
            print(f"빠른 추천 오류: {str(e)}")
            QMessageBox.warning(self, "추천 오류", f"추천 중 오류가 발생했습니다: {str(e)}")
            
             # 로그 파일에 오류 저장 25.05.01 추가 
            with open("err_log.txt", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"[{datetime.now()}] 빠른 추천 오류: \n{err_log.txt}\n\n")
                
    def show_health_recommendation(self):
        """건강 맞춤 추천 화면 표시"""
        self.clear_content()
        
        # 건강 맞춤 추천 타이틀
        title_label = QLabel("건강 맞춤 추천")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px 0;")
        
        # 연령대 선택
        age_frame = QFrame()
        age_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        age_layout = QVBoxLayout(age_frame)
        
        age_label = QLabel("연령대 선택")
        age_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        self.age_combo = QComboBox()
        self.age_combo.addItems(["선택 안함", "어린이", "청소년", "청년", "중년", "장년", "노년"])
        self.age_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
            }
        """)
        
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_combo)
        
        # 질환 선택
        disease_frame = QFrame()
        disease_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        disease_layout = QVBoxLayout(disease_frame)
        
        disease_label = QLabel("질환 정보 선택 (해당되는 항목 모두 선택)")
        disease_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        
        disease_layout.addWidget(disease_label)
        
        # 질환 체크박스
        self.disease_checks = {}
        for disease in ["당뇨병", "고혈압", "심장질환", "고지혈증"]:
            check = QCheckBox(disease)
            check.setStyleSheet("font-size: 14px;")
            disease_layout.addWidget(check)
            self.disease_checks[disease] = check
        
        # 추천 버튼
        recommend_btn = QPushButton("맞춤 추천 받기")
        recommend_btn.setStyleSheet("""
            QPushButton {
                background-color: #E54F40;
                color: white;
                border-radius: 10px;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #D44539;
            }
        """)
        recommend_btn.clicked.connect(self.get_health_recommendation)
        
        # 결과 섹션 (초기에는 숨김)
        self.result_frame = QFrame()
        self.result_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        self.result_frame.setVisible(False)
        
        result_layout = QVBoxLayout(self.result_frame)
        
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("font-size: 16px;")
        
        # 음성으로 들려주기 버튼
        self.result_tts_btn = QPushButton("음성으로 들려주기")
        self.result_tts_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.result_tts_btn.clicked.connect(lambda: self.play_tts(self.result_label.text()))
        
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.result_tts_btn)
        
        # 추천 메뉴 리스트 (초기에는 빈 상태)
        self.recommendation_list = QListWidget()
        self.recommendation_list.setStyleSheet("""
            QListWidget {
                background-color: #f5f5f5;
                border-radius: 10px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
            }
        """)
        self.recommendation_list.setVisible(False)
        
        # 장바구니 추가 버튼
        self.cart_add_btn = QPushButton("선택한 메뉴 장바구니에 추가")
        self.cart_add_btn.setStyleSheet("""
            QPushButton {
                background-color: #FFCC00;
                border-radius: 10px;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #E5B800;
            }
        """)
        self.cart_add_btn.clicked.connect(self.add_selected_to_cart)
        self.cart_add_btn.setVisible(False)
        
        # 컨텐츠 영역에 위젯 추가
        self.content_layout.addWidget(title_label)
        self.content_layout.addWidget(age_frame)
        self.content_layout.addWidget(disease_frame)
        self.content_layout.addWidget(recommend_btn)
        self.content_layout.addWidget(self.result_frame)
        self.content_layout.addWidget(self.recommendation_list)
        self.content_layout.addWidget(self.cart_add_btn)
        
        # 뒤로가기 버튼 추가
        self.add_back_button()
    
    def get_health_recommendation(self):
        """건강 맞춤 추천 함수"""
        try:
            # 선택된 연령대
            age_group = self.age_combo.currentText()
            if age_group == "선택 안함":
                age_group = None
            
            # 선택된 질환
            diseases = []
            for disease, checkbox in self.disease_checks.items():
                if checkbox.isChecked():
                    diseases.append(disease)
            
            # 선택 여부 확인
            if not age_group and not diseases:
                QMessageBox.warning(self, "선택 필요", "연령대나 질환을 하나 이상 선택해주세요.")
                return
            
            # preprocessing.py의 recommend_by_age_and_disease 함수 호출
            recommendation_text = recommend_by_age_and_disease(self.menu_df, age_group, diseases if diseases else None)
            
            # 결과 표시
            self.result_label.setText(recommendation_text)
            self.result_frame.setVisible(True)
            
            # 추천된 메뉴 추출
            menu_names = []
            if "맞춤 추천 메뉴입니다" in recommendation_text:
                menu_text = recommendation_text.split("맞춤 추천 메뉴입니다: ")[1].split(".")[0]
                menu_names = [name.strip() for name in menu_text.split(", ")]
            
            # 추천 메뉴 리스트 생성
            self.recommendation_list.clear()
            self.recommendation_checkboxes = []
            
            for name in menu_names:
                # 메뉴 가격 찾기
                price = 0
                for _, row in self.menu_df.iterrows():
                    if row['이름'] == name:
                        price = row['가격']
                        break
                
                item = QListWidgetItem()
                item_widget = QWidget()
                item_layout = QHBoxLayout(item_widget)
                
                checkbox = QCheckBox(f"{name} ({price}원)")
                checkbox.setStyleSheet("font-size: 14px;")
                
                item_layout.addWidget(checkbox)
                
                item.setSizeHint(item_widget.sizeHint())
                self.recommendation_list.addItem(item)
                self.recommendation_list.setItemWidget(item, item_widget)
                
                self.recommendation_checkboxes.append((checkbox, name, price))
            
            # 리스트 및 장바구니 버튼 표시
            self.recommendation_list.setVisible(True)
            self.cart_add_btn.setVisible(True)
        
        except Exception as e:
            print(f"건강 맞춤 추천 오류: {str(e)}")
            QMessageBox.warning(self, "추천 오류", f"추천 중 오류가 발생했습니다: {str(e)}")
    
    def show_chat_interface(self):
        """채팅 인터페이스 화면 표시"""
        self.clear_content()
        
        # 채팅 타이틀
        title_label = QLabel("메뉴에 대해 물어보세요")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px 0;")
        
        # 채팅 창
        chat_frame = QFrame()
        chat_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        chat_layout = QVBoxLayout(chat_frame)
        
        # 채팅 이력
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        # 시작 메시지
        welcome_msg = "AI 도우미: 안녕하세요! 메뉴에 대해 무엇이든 물어보세요. 예를 들어 '아메리카노는 얼마인가요?', '달달한 음료 추천해주세요' 등을 물어볼 수 있어요."
        self.chat_history.append(welcome_msg)
        
        # 질문 입력 영역
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("질문을 입력하세요...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        self.chat_input.returnPressed.connect(self.send_question)
        
        send_btn = QPushButton("전송")
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        send_btn.clicked.connect(self.send_question)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_btn)
        
        # 예시 질문
        example_frame = QFrame()
        example_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        example_layout = QVBoxLayout(example_frame)
        
        example_label = QLabel("이런 질문을 물어보세요:")
        example_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        example_layout.addWidget(example_label)
        
        # 예시 질문 버튼들
        example_questions = [
            "아메리카노와 카페라떼 중 어떤 게 더 달아요?",
            "디카페인 커피는 어떤 게 있나요?",
            "아이스 음료 중에 단 게 뭐가 있을까요?",
            "따뜻한 음료를 추천해주세요."
        ]
        
        for question in example_questions:
            btn = QPushButton(question)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    padding: 5px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            btn.clicked.connect(lambda _, q=question: self.ask_example_question(q))
            example_layout.addWidget(btn)
        
        # 채팅 레이아웃에 위젯 추가
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(input_frame)
        
        # 컨텐츠 영역에 위젯 추가
        self.content_layout.addWidget(title_label)
        self.content_layout.addWidget(chat_frame)
        self.content_layout.addWidget(example_frame)
        
        # 뒤로가기 버튼 추가
        self.add_back_button()
    
    def send_question(self):
        """질문 전송"""
        question = self.chat_input.text().strip()
        if not question:
            return
        
        # 질문 표시
        self.chat_history.append(f"\n나: {question}")
        
        # 입력창 초기화
        self.chat_input.clear()
        
        # 25.05.01 대화 이력 초기화 (없으면)
        if not hasattr(self, 'conversation_history'):
            self.conversation_history = {
                'pending_menu': None,
                'previous_query': None,
                'previous_response': None
            }
    
        # 대화 이력 업데이트
        self.conversation_history['previous_query'] = question

        # RAG 파이프라인을 통한 답변 생성
         
            # preprocessing.py의 rag_pipeline 함수 호출
            #answer = rag_pipeline(question, self.menu_texts, self.embedder, self.index)
            
        try:
            # preprocessing.py의 rag_pipeline 함수 호출
            answer = rag_pipeline(
                question, 
                self.menu_texts, 
                self.embedder, 
                self.index, 
                menu_df=self.menu_df,
                conversation_history=self.conversation_history
            )

             # 대화 이력 응답 업데이트
            self.conversation_history['previous_response'] = answer

            # 답변 표시
            self.chat_history.append(f"\nAI 도우미: {answer}")
            
            # 채팅 창 스크롤
            self.chat_history.moveCursor(self.chat_history.textCursor().End)
        
        except Exception as e:
            # print(f"RAG 파이프라인 오류: {str(e)}")
            # self.chat_history.append("\nAI 도우미: 죄송합니다, 질문에 답변하는 중 오류가 발생했습니다.")
            import traceback
            error_msg = traceback.format_exc()
            print(f"RAG 파이프라인 오류: {str(e)}")
            print(error_msg)
            self.chat_history.append("\nAI 도우미: 죄송합니다, 질문에 답변하는 중 오류가 발생했습니다.")
            
            # 로그 파일에 오류 저장
            with open("err_log.txt", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"[{datetime.now()}] RAG 파이프라인 오류: \n{error_msg}\n\n")
    
    def ask_example_question(self, question):
        """예시 질문 클릭"""
        self.chat_input.setText(question)
        self.send_question()
    
    def add_selected_to_cart(self):
        """선택한 추천 메뉴를 장바구니에 추가"""
        selected_menus = []
        
        # 선택된 체크박스 확인
        for checkbox, name, price in self.recommendation_checkboxes:
            if checkbox.isChecked():
                selected_menus.append((name, price))
        
        if not selected_menus:
            QMessageBox.warning(self, "선택 필요", "장바구니에 추가할 메뉴를 선택해주세요.")
            return
        
        # 부모 윈도우의 장바구니에 추가
        for name, price in selected_menus:
            self.parent.add_to_cart(name, price)
        
        QMessageBox.information(self, "장바구니 추가", "선택한 메뉴가 장바구니에 추가되었습니다.")
    
    def play_tts(self, text):
        """TTS로 텍스트를 읽어주는 함수"""
        try:
            # preprocessing.py의 text_to_speech 함수 호출
            text_to_speech(text)
        except Exception as e:
            print(f"TTS 오류: {str(e)}")
            QMessageBox.warning(self, "TTS 오류", "음성 변환 중 오류가 발생했습니다.")

# 메인 함수
def main():
    app = QApplication(sys.argv)
    
    # 폰트 설정
    app.setFont(QFont("맑은 고딕", 10))
    
    # 메인 윈도우 생성 및 표시
    window = MegaKioskUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()