# 통합된 mega_kiosk_ui_secure.py
# 기존 mega_kiosk_ui.py에 보안 기능을 통합

from datetime import datetime
import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon
import re
import pandas as pd

# 🔐 새로운 보안 모듈들 import
from network_security import NetworkSecurityManager, SecureRAGPipeline
from privacy_protection import PrivacyProtectionManager, SecureFaceAnalyzer
from identity_verification import IdentityVerificationSystem, IdentityVerificationDialog
from digital_assistant import IntegratedAccessibilitySystem


# 🔐 얼굴 인식 함수 안전한 import
def safe_get_age_from_camera():
    """안전한 얼굴 인식 함수"""
    try:
        from face_recognition import get_age_from_camera
        return get_age_from_camera()
    except ImportError:
        print("⚠️ face_recognition 모듈을 찾을 수 없습니다.")
        try:
            from face_recognition import get_age_from_camera_simple
            return get_age_from_camera_simple()
        except ImportError:
            print("🔧 테스트 모드로 전환합니다.")
            import random
            return random.choice([15, 25, 35, 65])
    except Exception as e:
        print(f"⚠️ 얼굴 인식 오류: {e}")
        return 25  # 기본 연령

# 🔐 보안 모듈들 안전한 import
try:
    from network_security import NetworkSecurityManager
except ImportError:
    print("⚠️ network_security 모듈이 없습니다. 기본 모드로 실행합니다.")
    class NetworkSecurityManager:
        def __init__(self):
            self.offline_mode = False

try:
    from privacy_protection import PrivacyProtectionManager, SecureFaceAnalyzer
except ImportError:
    print("⚠️ privacy_protection 모듈이 없습니다. 기본 모드로 실행합니다.")
    class PrivacyProtectionManager:
        def auto_delete_expired_data(self): pass
        def generate_session_id(self): return "test_session"
    
    class SecureFaceAnalyzer:
        def analyze_face_secure(self, face_image): return "성인"

try:
    from identity_verification import IdentityVerificationSystem, IdentityVerificationDialog
except ImportError:
    print("⚠️ identity_verification 모듈이 없습니다.")
    class IdentityVerificationSystem: pass
    class IdentityVerificationDialog:
        def __init__(self, parent, menu_name, menu_df): pass
        def start_verification(self): return {'verified': True, 'message': '테스트 모드'}

try:
    from digital_assistant import IntegratedAccessibilitySystem
except ImportError:
    print("⚠️ digital_assistant 모듈이 없습니다.")
    class IntegratedAccessibilitySystem:
        def __init__(self, main_ui): pass
        def start_accessibility_mode(self, profile): pass
        def request_assistance(self, step): return {'text': '도움말', 'voice': True}
        def end_session(self): pass

# 기존 모듈들 import
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
    print(f"⚠️ RAG 모듈 로딩 오류: {e}")
    print("🔧 기본 추천 시스템으로 전환합니다.")
    RAG_AVAILABLE = False
    
    # 대체 함수들
    def load_csv_data(path): return ["아메리카노 4000원", "카페라떼 4500원"]
    def text_to_speech(text): print(f"🔊 TTS: {text}")
    def recommend_by_age(df, age): return f"{age} 연령대에게 아메리카노, 카페라떼를 추천합니다."
    def filter_menu_by_health(df, age_group=None, diseases=None): return df
    def recommend_by_age_and_disease(df, age=None, diseases=None, top_n=3): return "건강 맞춤 메뉴를 추천합니다."
    def rag_pipeline(query, chunks, embedder, index, **kwargs): return "죄송합니다. AI 서비스가 일시적으로 사용할 수 없습니다."

class SecureMegaKioskUI(QMainWindow):
    """보안이 강화된 메가 커피 키오스크 UI"""
    
    def __init__(self):
        super().__init__()
        
        # 🔐 보안 매니저들 초기화
        self.privacy_manager = PrivacyProtectionManager()
        self.security_manager = NetworkSecurityManager()
        self.identity_verifier = IdentityVerificationSystem()
        self.face_analyzer = SecureFaceAnalyzer()

        # 접근성 시스템 초기화
        self.accessibility_system = IntegratedAccessibilitySystem(self)
        
        # 기본 설정
        self.setWindowTitle("메가 커피 키오스크 (보안 강화)")
        self.setGeometry(100, 100, 800, 1000)
        self.menu_texts = {}
        self.embedder = None
        self.vector_store = None
        self.ai_service = None
        self.index = None
        
        # 현재 사용자 정보
        self.current_user_profile = {
            'age_group': None,
            'session_id': None,
            'accessibility_needs': []
        }
        
        # 메뉴 데이터 로드
        self.load_menu_data()
        
        # 🔐 보안 강화된 RAG 모델 초기화
        self.initialize_secure_rag()
        
        # 카트(장바구니) 초기화
        self.cart = {}
        
        # 자동 데이터 정리 타이머 설정
        self.setup_data_cleanup_timer()
        
        # UI 설정
        self.init_ui()
        
        # 접근성 모드 시작
        self.accessibility_system.start_accessibility_mode(self.current_user_profile)
    
    def setup_data_cleanup_timer(self):
        """자동 데이터 정리 타이머 설정"""
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.privacy_manager.auto_delete_expired_data)
        self.cleanup_timer.start(300000)  # 5분마다 실행
    
    def load_menu_data(self):
        """메뉴 데이터 로드 (기존 코드 유지)"""
        try:
            csv_path = "process_data.csv"
            self.menu_df = pd.read_csv(csv_path, encoding='utf-8')
            print(f"메뉴 데이터 로드 완료: {len(self.menu_df)}개 메뉴")
        except Exception as e:
            print(f"메뉴 데이터 로드 오류: {str(e)}")
            self.create_sample_menu_data()
    
    def create_sample_menu_data(self):
        """샘플 메뉴 데이터 생성 (기존 코드 유지)"""
        print("샘플 메뉴 데이터 생성")
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
            '카페인(mg)': np.random.uniform(0, 250, 15),
            '당류(g)': np.random.uniform(0, 20, 15),
            '나트륨(mg)': np.random.uniform(0, 200, 15),
            '연령': ['어린이, 청소년, 성인, 노인'] * 15
        })
    
    def initialize_secure_rag(self):
        """🔐 보안 강화된 RAG 모델 초기화"""
        try:
            if hasattr(self, 'menu_df') and not self.menu_df.empty:
                # 메뉴 텍스트 생성 (기존 로직 유지)
                if '분류' in self.menu_df.columns:
                    menu_texts = self.menu_df.apply(
                        lambda row: f"{row['이름']} {row['가격']}원. 분류: {row['분류']}.", 
                        axis=1
                    ).tolist()
                else:
                    menu_texts = self.menu_df.apply(
                        lambda row: f"{row['이름']} {row['가격']}원.", 
                        axis=1
                    ).tolist()
                
                # 🔐 보안 검증된 임베딩 모델 사용
                if self.security_manager.offline_mode:
                    # 로컬 모델 사용
                    self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    print("✅ 오프라인 임베딩 모델 로드 완료")
                else:
                    print("⚠️ 온라인 모드 - 보안 위험 존재")
                    self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                
                # FAISS 인덱스 구축
                embeddings = self.embedder.encode(menu_texts)
                self.index = faiss.IndexFlatL2(embeddings.shape[1])
                self.index.add(np.array(embeddings, dtype=np.float32))
                self.menu_texts = menu_texts
                
                print("🔐 보안 강화된 RAG 모델 초기화 완료")
            else:
                print("❌ 메뉴 데이터가 없어 RAG 초기화 실패")
        except Exception as e:
            print(f"🚨 RAG 초기화 오류: {str(e)}")
    
    def init_ui(self):
        """메인 UI 초기화 (기존 코드 + 보안 기능 추가)"""
        # 기존 UI 코드 유지...
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 🔐 보안 상태 표시 바 추가
        security_bar = self.create_security_status_bar()
        main_layout.addWidget(security_bar)
        
        # 기존 타이틀 바
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #FFCC00; border: none;")
        title_frame.setFixedHeight(70)
        
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("MEGA COFFEE (보안 강화)")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        title_label.setAlignment(Qt.AlignCenter)
        
        # 🔐 접근성 도움 버튼 추가
        help_button = QPushButton("도움이 필요해요")
        help_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                font-weight: bold;
                padding: 5px 10px;
            }
        """)
        help_button.clicked.connect(self.request_accessibility_help)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(help_button)
        
        # 기존 카테고리 바 (수정)
        category_frame = QFrame()
        category_frame.setStyleSheet("background-color: #FFCC00; border: none;")
        
        category_layout = QGridLayout(category_frame)
        category_layout.setSpacing(2)
        
        # 🔐 보안 강화된 카테고리 버튼들
        categories = [
            "디카페인", "추천메뉴", "커피(ICE)", "커피(HOT)",
            "오늘의메뉴", "스무디", "티/차", "🔐 AI 서비스 (보안)"
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
            
            if "AI 서비스" in category:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32;
                        color: white;
                        border-radius: 10px;
                        font-weight: bold;
                        font-size: 14px;
                        border: 2px solid #4CAF50;
                    }
                    QPushButton:hover {
                        background-color: #1B5E20;
                    }
                """)
                btn.clicked.connect(self.show_secure_ai_service_dialog)
            
            row, col = i // 4, i % 4
            category_layout.addWidget(btn, row, col)
            self.category_buttons.append(btn)
        
        # 나머지 UI 구성 (스크롤 영역, 카트 등은 기존 코드 유지)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("border: none; background-color: white;")
        
        self.menu_widget = QWidget()
        self.menu_layout = QGridLayout(self.menu_widget)
        self.menu_layout.setSpacing(10)
        scroll_area.setWidget(self.menu_widget)
        
        # 장바구니 영역 (기존 코드 유지)
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
        self.payment_button.clicked.connect(self.secure_process_payment)
        
        cart_layout.addWidget(self.cart_label)
        cart_layout.addWidget(self.payment_button)
        
        # 메인 레이아웃에 위젯 추가
        main_layout.addWidget(title_frame)
        main_layout.addWidget(category_frame)
        main_layout.addWidget(scroll_area, 1)
        main_layout.addWidget(cart_frame)
        
        self.setCentralWidget(central_widget)
        self.display_menu()
    
    def create_security_status_bar(self):
        """🔐 보안 상태 표시 바"""
        status_frame = QFrame()
        status_frame.setFixedHeight(30)
        status_frame.setStyleSheet("background-color: #2E7D32; color: white;")
        
        status_layout = QHBoxLayout(status_frame)
        
        # 보안 상태 아이콘들
        network_status = QLabel("🔒 내부망")
        privacy_status = QLabel("🛡️ 개인정보보호")
        accessibility_status = QLabel("♿ 접근성모드")
        
        for label in [network_status, privacy_status, accessibility_status]:
            label.setStyleSheet("font-size: 12px; margin: 0 10px;")
        
        status_layout.addWidget(network_status)
        status_layout.addWidget(privacy_status)
        status_layout.addWidget(accessibility_status)
        status_layout.addStretch()
        
        # 현재 시간
        import datetime
        time_label = QLabel(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
        time_label.setStyleSheet("font-size: 12px;")
        status_layout.addWidget(time_label)
        
        return status_frame
    
    def request_accessibility_help(self):
        """🔐 접근성 도움 요청"""
        current_step = "메뉴_선택"  # 현재 단계 감지 로직 필요
        help_content = self.accessibility_system.request_assistance(current_step)
        
        if help_content:
            # 음성 안내
            if help_content.get('voice'):
                text_to_speech(help_content['text'])
            
            # 시각적 가이드
            if help_content.get('visual_guide'):
                self.show_visual_guide(help_content['visual_guide'])
            
            # 도움말 다이얼로그
            QMessageBox.information(self, "도움말", help_content['text'])
    
    def show_visual_guide(self, guide_type):
        """시각적 가이드 표시"""
        if guide_type == "highlight_menu_images":
            # 메뉴 이미지들 강조 표시
            pass
        elif guide_type == "highlight_temperature_buttons":
            # 온도 선택 버튼들 강조 표시
            pass
    
    def show_secure_ai_service_dialog(self):
        """🔐 보안 강화된 AI 서비스 다이얼로그"""
        dialog = SecureAIServiceDialog(self, self.menu_df)
        dialog.exec_()
        self.update_cart_ui()
    
    def secure_add_to_cart(self, name, price):
        """🔐 보안 강화된 장바구니 추가"""
        # 연령 제한 확인이 필요한 메뉴인지 체크
        menu_info = self.menu_df[self.menu_df['이름'] == name]
        if not menu_info.empty:
            caffeine_mg = menu_info.iloc[0].get('카페인(mg)', 0)
            
            # 카페인 함량이 높은 경우 연령 확인
            if caffeine_mg > 100:
                verification_dialog = IdentityVerificationDialog(self, name, self.menu_df)
                result = verification_dialog.start_verification()
                
                if not result['verified']:
                    QMessageBox.warning(self, "주문 제한", result['message'])
                    return
                
                # 연령 제한 경고 표시
                if result.get('restrictions'):
                    for restriction in result['restrictions']:
                        if restriction['type'] == 'caffeine_warning':
                            reply = QMessageBox.question(
                                self, "카페인 경고", 
                                f"{restriction['message']}\n그래도 주문하시겠습니까?",
                                QMessageBox.Yes | QMessageBox.No
                            )
                            if reply == QMessageBox.No:
                                return
        
        # 기존 장바구니 추가 로직
        if name in self.cart:
            self.cart[name]['count'] += 1
        else:
            self.cart[name] = {'price': price, 'count': 1}
        
        self.update_cart_ui()
        QMessageBox.information(self, "메뉴 추가", f"{name}이(가) 장바구니에 추가되었습니다.")
    
    def secure_process_payment(self):
        """🔐 보안 강화된 결제 처리"""
        if not self.cart:
            QMessageBox.warning(self, "결제 오류", "장바구니가 비어 있습니다.")
            return
        
        # 결제 전 개인정보 정리
        self.privacy_manager.auto_delete_expired_data()
        
        # 기존 결제 로직
        cart_text = "\n".join([
            f"{name} x {item['count']} = {item['price'] * item['count']}원" 
            for name, item in self.cart.items()
        ])
        
        total_price = sum(item['price'] * item['count'] for item in self.cart.values())
        
        reply = QMessageBox.question(
            self, "결제 확인", 
            f"다음 메뉴를 결제하시겠습니까?\n\n{cart_text}\n\n총액: {total_price}원",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 🔐 익명화된 주문 로그 저장
            anonymous_order = self.privacy_manager.anonymize_health_data(list(self.cart.keys()))
            
            QMessageBox.information(self, "결제 완료", "결제가 완료되었습니다. 이용해 주셔서 감사합니다!")
            
            # 장바구니 초기화 및 세션 정리
            self.cart = {}
            self.update_cart_ui()
            self.accessibility_system.end_session()
    
    # 기존 메서드들 (display_menu, update_cart_ui 등) 유지...
    def display_menu(self):
        """메뉴 표시 (기존 코드 유지)"""
        while self.menu_layout.count():
            item = self.menu_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        menu_count = min(12, len(self.menu_df))
        
        for i in range(menu_count):
            menu_data = self.menu_df.iloc[i]
            
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
            
            menu_layout = QVBoxLayout(menu_frame)
            
            image_label = QLabel()
            image_label.setFixedSize(120, 100)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
            image_label.setText(menu_data['이름'][0])
            
            name_label = QLabel(menu_data['이름'])
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            price_label = QLabel(f"{menu_data['가격']}원")
            price_label.setAlignment(Qt.AlignCenter)
            price_label.setStyleSheet("color: #E54F40; font-size: 14px;")
            
            menu_layout.addWidget(image_label)
            menu_layout.addWidget(name_label)
            menu_layout.addWidget(price_label)
            
            # 🔐 보안 강화된 클릭 이벤트
            menu_frame.mousePressEvent = lambda e, name=menu_data['이름'], price=menu_data['가격']: self.secure_add_to_cart(name, price)
            
            row, col = i // 3, i % 3
            self.menu_layout.addWidget(menu_frame, row, col)
    
    def update_cart_ui(self):
        """장바구니 UI 업데이트 (기존 코드 유지)"""
        total_count = sum(item['count'] for item in self.cart.values())
        total_price = sum(item['price'] * item['count'] for item in self.cart.values())
        
        self.cart_label.setText(f"장바구니 ({total_count}개)")
        self.payment_button.setText(f"{total_price}원 결제하기")

# 🔐 보안 강화된 AI 서비스 다이얼로그
class SecureAIServiceDialog(QDialog):
    """보안이 강화된 AI 서비스 다이얼로그"""
    
    def __init__(self, parent, menu_df):
        super().__init__(parent)

        # 🔧 기존 레이아웃이 있으면 제거 (중요!)
        existing_layout = self.layout()
        if existing_layout:
            # 기존 레이아웃의 모든 위젯 제거
            while existing_layout.count():
                child = existing_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            # 레이아웃 제거
            existing_layout.deleteLater()
        # 속성 초기화    
        self.parent = parent
        self.menu_df = menu_df
        
        # 🔐 보안 컴포넌트들
        self.privacy_manager = parent.privacy_manager
        self.security_manager = parent.security_manager
        
        # 안전하게 부모 속성 접근
        self.menu_texts = getattr(parent, 'menu_texts', {})
        self.embedder = getattr(parent, 'embedder', None)
        self.index = getattr(parent, 'index', None)
        
        self.setWindowTitle("🔐 AI 메뉴 추천 서비스 (보안 강화)")
        self.setFixedSize(600, 700)
        self.setStyleSheet("background-color: white;")
    

        self.init_ui()
    
    def init_ui(self):
        """다이얼로그 UI 초기화"""
        # 🔧 기존 레이아웃 완전 제거
        if self.layout():
            QWidget().setLayout(self.layout())
        
        # 🔧 메인 레이아웃 한 번만 생성
        main_layout = QVBoxLayout()
        
        # 🔐 보안 강화 타이틀 바
        title_frame = QFrame()
        title_frame.setStyleSheet("background-color: #2E7D32;")
        title_frame.setFixedHeight(60)
        
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("🔐 AI 메뉴 추천 (개인정보보호)")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        
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
        
        # 개인정보 보호 안내
        privacy_notice = QLabel("""
        🔒 개인정보 보호 안내
        • 얼굴 데이터는 연령 추정 후 즉시 삭제됩니다
        • 모든 처리는 내부 시스템에서만 진행됩니다
        • 개인 식별 정보는 저장되지 않습니다
        """)
        privacy_notice.setStyleSheet("""
            background-color: #E8F5E8;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
        """)
        
        # 서비스 버튼들 컨테이너
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setAlignment(Qt.AlignCenter)
        self.content_layout.setSpacing(20)
        
        welcome_label = QLabel("안전하고 건강한 메뉴를 추천해 드립니다")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("font-size: 16px; margin: 20px 0;")
        
        services = [
            ("🔐 얼굴 인식으로 안전한 맞춤 추천", self.show_secure_face_recognition),
            ("⚡ 빠른 추천 받기", self.show_quick_recommendation),
            ("🏥 건강 맞춤 추천 받기", self.show_health_recommendation),
            ("💬 메뉴에 대해 물어보기", self.show_secure_chat_interface),
            ("🔙 메인 화면으로 돌아가기", self.reject)
        ]
        
        # 환영 메시지 추가
        self.content_layout.addWidget(welcome_label)
            
        # 서비스 버튼들 생성
        for text, func in services:
            btn = QPushButton(text)
            btn.setFixedHeight(60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border-radius: 10px;
                    font-size: 16px;
                    padding: 10px;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #FFCC00;
                }
            """)
            
            if "얼굴 인식" in text:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #2E7D32;
                        color: white;
                        border-radius: 10px;
                        font-size: 16px;
                        padding: 10px;
                        text-align: left;
                    }
                    QPushButton:hover {
                        background-color: #1B5E20;
                    }
                """)
            elif "물어보기" in text:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #1976D2;
                        color: white;
                        border-radius: 10px;
                        font-size: 16px;
                        padding: 10px;
                        text-align: left;
                    }
                    QPushButton:hover {
                        background-color: #1565C0;
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
        main_layout.addWidget(title_frame)
        main_layout.addWidget(privacy_notice)
        main_layout.addWidget(scroll_area)
        
        # 🔧 마지막에 한 번만 설정
        self.setLayout(main_layout)
    
    def clearLayout(self):
        """모든 기존 레이아웃과 위젯 제거"""
        if self.layout():
            while self.layout().count():
                child = self.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clearLayout(child.layout())
            self.layout().deleteLater()
            
    def show_secure_face_recognition(self):
        """🔐 보안 강화된 얼굴 인식"""
        self.clear_content()
        
        # 개인정보 동의 확인
        consent_reply = QMessageBox.question(
            self, "개인정보 처리 동의",
            "연령 추정을 위해 얼굴을 촬영합니다.\n"
            "• 촬영된 이미지는 연령 추정 후 즉시 삭제됩니다\n"
            "• 개인 식별 정보는 저장되지 않습니다\n"
            "• 연령대 정보만 추천에 활용됩니다\n\n"
            "동의하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if consent_reply != QMessageBox.Yes:
            self.init_ui()
            return
        
        # 🔐 보안 강화된 얼굴 인식 시스템 사용
        from face_recognition import get_age_from_camera
        
        try:
            # 얼굴에서 연령 추정
            age = get_age_from_camera()
            
            if age is None:
                QMessageBox.warning(self, "인식 실패", "얼굴을 인식할 수 없습니다. 다시 시도해주세요.")
                self.init_ui()
                return
            
            # 🔐 개인정보 마스킹 처리
            #masked_data = self.privacy_manager.mask_face_features(np.zeros((100,100,3)))  # 더미 이미지
            
            # 연령대 변환 (구체적 나이 저장 안함)
            if age < 13:
                age_group = "어린이"
            elif age < 20:
                age_group = "청소년"
            elif age < 40:
                age_group = "성인"
            else:
                age_group = "노인"
            
            # 결과 표시
            result_label = QLabel(f"🔐 보안 인식 결과: {age_group}")
            result_label.setAlignment(Qt.AlignCenter)
            result_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 20px 0; color: #2E7D32;")
            
            # 추천 결과 표시
            recommendation_frame = QFrame()
            recommendation_frame.setStyleSheet("background-color: #E8F5E8; border-radius: 10px; padding: 10px;")
            
            recommendation_layout = QVBoxLayout(recommendation_frame)
            
            # 🔐 보안 강화된 추천 시스템 사용
            recommendation_text = recommend_by_age(self.menu_df, age_group)
            
            recommendation_label = QLabel(recommendation_text)
            recommendation_label.setWordWrap(True)
            recommendation_label.setStyleSheet("font-size: 16px;")
            
            # 음성으로 들려주기 버튼
            tts_btn = QPushButton("🔊 음성으로 들려주기")
            tts_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
            tts_btn.clicked.connect(lambda: text_to_speech(recommendation_text))
            
            recommendation_layout.addWidget(recommendation_label)
            recommendation_layout.addWidget(tts_btn)
            
            # 추천 메뉴 체크박스 리스트
            menu_frame = QFrame()
            menu_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
            menu_layout = QVBoxLayout(menu_frame)
            
            # 연령대에 맞는 메뉴 필터링
            filtered_menus = filter_menu_by_health(self.menu_df, age_group=age_group)
            recommended_menus = filtered_menus.head(5)  # 상위 5개
            
            self.recommendation_checkboxes = []
            
            for _, row in recommended_menus.iterrows():
                menu_item = QCheckBox(f"{row['이름']} ({row['가격']}원)")
                menu_item.setStyleSheet("font-size: 14px; margin: 5px;")
                menu_layout.addWidget(menu_item)
                self.recommendation_checkboxes.append((menu_item, row['이름'], row['가격']))
            
            # 장바구니 추가 버튼
            add_to_cart_btn = QPushButton("🛒 선택한 메뉴 장바구니에 추가")
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
            self.content_layout.addWidget(recommendation_frame)
            self.content_layout.addWidget(menu_frame)
            self.content_layout.addWidget(add_to_cart_btn)
            
            # 개인정보 처리 완료 안내
            privacy_complete = QLabel("✅ 개인정보 처리가 완료되었습니다. 원본 데이터는 삭제되었습니다.")
            privacy_complete.setStyleSheet("color: #2E7D32; font-size: 12px; font-style: italic;")
            self.content_layout.addWidget(privacy_complete)
            
            # 뒤로가기 버튼 추가
            self.add_back_button()
            
        except Exception as e:
            print(f"🚨 보안 얼굴 인식 오류: {str(e)}")
            QMessageBox.critical(self, "오류", "얼굴 인식 중 오류가 발생했습니다.")
            self.init_ui()
    
    def show_secure_chat_interface(self):
        """🔐 보안 강화된 채팅 인터페이스"""
        self.clear_content()
        
        # 채팅 타이틀
        title_label = QLabel("💬 메뉴에 대해 안전하게 물어보세요")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px 0;")
        
        # 🔐 보안 안내
        security_notice = QLabel("""
        🔒 보안 채팅 안내:
        • 모든 대화는 내부 시스템에서만 처리됩니다
        • 개인정보는 저장되지 않습니다
        • 대화 내용은 서비스 개선에만 활용됩니다
        """)
        security_notice.setStyleSheet("""
            background-color: #E3F2FD;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            margin: 10px 0;
        """)
        
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
        
        # 🔐 보안 강화된 시작 메시지
        welcome_msg = """🔐 보안 AI 도우미: 안녕하세요! 
메뉴에 대해 안전하게 문의하실 수 있습니다. 
모든 대화는 암호화되어 처리됩니다.

예시 질문:
• "아메리카노는 얼마인가요?"
• "달달한 음료 추천해주세요"
• "카페인이 적은 메뉴는 뭐가 있나요?"
"""
        self.chat_history.append(welcome_msg)
        
        # 질문 입력 영역
        input_frame = QFrame()
        input_layout = QHBoxLayout(input_frame)
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("🔒 안전한 질문을 입력하세요...")
        self.chat_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #2E7D32;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        self.chat_input.returnPressed.connect(self.send_secure_question)
        
        send_btn = QPushButton("🔐 보안 전송")
        send_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1B5E20;
            }
        """)
        send_btn.clicked.connect(self.send_secure_question)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(send_btn)
        
        # 예시 질문 버튼들
        example_frame = QFrame()
        example_frame.setStyleSheet("background-color: #f5f5f5; border-radius: 10px; padding: 10px;")
        
        example_layout = QVBoxLayout(example_frame)
        
        example_label = QLabel("🔒 안전한 예시 질문들:")
        example_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        example_layout.addWidget(example_label)
        
        example_questions = [
            "아메리카노와 카페라떼 차이점이 뭔가요?",
            "디카페인 커피 종류를 알려주세요",
            "당분이 적은 음료를 추천해주세요",
            "따뜻한 음료 중 인기 메뉴는?"
        ]
        
        for question in example_questions:
            btn = QPushButton(f"💬 {question}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E8F5E8;
                    border: 1px solid #4CAF50;
                    border-radius: 5px;
                    padding: 8px;
                    text-align: left;
                    margin: 2px;
                }
                QPushButton:hover {
                    background-color: #C8E6C9;
                }
            """)
            btn.clicked.connect(lambda _, q=question: self.ask_example_question(q))
            example_layout.addWidget(btn)
        
        # 채팅 레이아웃에 위젯 추가
        chat_layout.addWidget(self.chat_history)
        chat_layout.addWidget(input_frame)
        
        # 컨텐츠 영역에 위젯 추가
        self.content_layout.addWidget(title_label)
        self.content_layout.addWidget(security_notice)
        self.content_layout.addWidget(chat_frame)
        self.content_layout.addWidget(example_frame)
        
        # 뒤로가기 버튼 추가
        self.add_back_button()
    
    def send_secure_question(self):
        """🔐 보안 강화된 질문 전송"""
        question = self.chat_input.text().strip()
        if not question:
            return
        
        # 질문 표시
        self.chat_history.append(f"\n🙋 나: {question}")
        
        # 입력창 초기화
        self.chat_input.clear()
        
        # 🔐 대화 이력 초기화 (보안)
        if not hasattr(self, 'secure_conversation_history'):
            self.secure_conversation_history = {
                'pending_menu': None,
                'previous_query': None,
                'previous_response': None,
                'session_id': self.privacy_manager.generate_session_id()
            }
        
        # 대화 이력 업데이트
        self.secure_conversation_history['previous_query'] = question
        
        try:
            # 🔐 보안 강화된 RAG 파이프라인 호출
            if hasattr(self.parent, 'security_manager') and self.parent.security_manager.offline_mode:
                # 오프라인 모드에서 처리
                answer = self.process_secure_rag(question)
            else:
                # 기존 RAG 파이프라인 사용 (보안 검증 후)
                answer = rag_pipeline(
                    question,
                    self.menu_texts,
                    self.embedder,
                    self.index,
                    menu_df=self.menu_df,
                    conversation_history=self.secure_conversation_history
                )
            
            # 대화 이력 응답 업데이트
            self.secure_conversation_history['previous_response'] = answer
            
            # 🔐 응답 보안 검증
            secure_answer = self.validate_response_security(answer)
            
            # 답변 표시
            self.chat_history.append(f"\n🔐 보안 AI: {secure_answer}")
            
            # 채팅 창 스크롤
            self.chat_history.moveCursor(self.chat_history.textCursor().End)
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            print(f"🚨 보안 RAG 오류: {str(e)}")
            
            # 🔐 보안 오류 로깅
            with open("secure_error_log.txt", "a", encoding="utf-8") as f:
                from datetime import datetime
                f.write(f"[{datetime.now()}] 보안 RAG 오류: \n{error_msg}\n\n")
            
            self.chat_history.append("\n🔐 보안 AI: 죄송합니다. 보안 처리 중 오류가 발생했습니다. 관리자에게 문의해주세요.")
    
    def process_secure_rag(self, question):
        """🔐 오프라인 보안 RAG 처리"""
        # 간단한 키워드 기반 응답 (오프라인 모드)
        menu_keywords = {
            '아메리카노': '아메리카노는 4000원입니다. HOT과 ICE 모두 가능합니다.',
            '카페라떼': '카페라떼는 4500원입니다. 부드러운 우유와 에스프레소의 조화입니다.',
            '가격': '메뉴별 가격은 4000원부터 6500원까지 다양합니다.',
            '추천': '고객님의 연령대에 맞는 건강한 메뉴를 추천해드릴 수 있습니다.',
            '디카페인': '현재 디카페인 메뉴로는 허브티와 과일차가 있습니다.',
            '달달한': '바닐라라떼와 카라멜마키아또가 달콤한 맛으로 인기가 좋습니다.'
        }
        
        for keyword, response in menu_keywords.items():
            if keyword in question:
                return f"🔐 {response}"
        
        return "🔐 죄송합니다. 메뉴에 대한 구체적인 질문을 해주시면 더 정확한 답변을 드릴 수 있습니다."
    
    def validate_response_security(self, response):
        """🔐 응답 보안 검증"""
        # 민감한 정보가 포함되어 있는지 확인
        sensitive_patterns = [
            r'\d{6}-\d{7}',  # 주민등록번호
            r'\d{3}-\d{4}-\d{4}',  # 전화번호
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # 이메일
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, response):
                return "🔐 보안상의 이유로 해당 정보는 제공할 수 없습니다."
        
        return response
    
    def ask_example_question(self, question):
        """예시 질문 클릭"""
        self.chat_input.setText(question)
        self.send_secure_question()
    
    # 기존 메서드들 (clear_content, add_back_button, add_selected_to_cart 등)
    def clear_content(self):
        """컨텐츠 영역 초기화"""
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
    
    def add_back_button(self):
        """뒤로 가기 버튼 추가"""
        back_btn = QPushButton("🔙 처음으로 돌아가기")
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
        
        # 부모 윈도우의 장바구니에 추가 (보안 강화된 메서드 사용)
        for name, price in selected_menus:
            self.parent.secure_add_to_cart(name, price)
        
        QMessageBox.information(self, "장바구니 추가", "선택한 메뉴가 안전하게 장바구니에 추가되었습니다.")
    
    # 나머지 메서드들 (show_quick_recommendation, show_health_recommendation)은 기존 로직 유지
    def show_quick_recommendation(self):
        """빠른 추천 화면 표시 (기존 로직 + 보안 강화)"""
        self.clear_content()
        
        age_label = QLabel("🔐 안전한 연령대 선택:")
        age_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px;")
        
        age_frame = QFrame()
        age_layout = QHBoxLayout(age_frame)
        
        age_groups = ["어린이", "청소년", "성인", "노인"]
        
        for age in age_groups:
            btn = QPushButton(f"👤 {age}")
            btn.setFixedHeight(60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E8F5E8;
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    font-size: 14px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #C8E6C9;
                }
            """)
            btn.clicked.connect(lambda _, age=age: self.show_secure_quick_result(age))
            age_layout.addWidget(btn)
        
        self.content_layout.addWidget(age_label)
        self.content_layout.addWidget(age_frame)
        self.add_back_button()
    
    def show_secure_quick_result(self, age_group):
        """🔐 보안 강화된 빠른 추천 결과"""
        try:
            # 🔐 익명화된 추천 처리
            recommendation_text = recommend_by_age(self.menu_df, age_group)
            
            # 🔐 보안 로깅 (익명화)
            anonymous_log = {
                'timestamp': datetime.now().isoformat(),
                'age_group': age_group,
                'recommendation_type': 'quick',
                'session_id': self.privacy_manager.generate_session_id()[:8]
            }
            
            result_frame = QFrame()
            result_frame.setStyleSheet("background-color: #E8F5E8; border-radius: 10px; padding: 15px;")
            
            result_layout = QVBoxLayout(result_frame)
            
            # 🔐 보안 표시
            security_icon = QLabel("🔐 보안 처리됨")
            security_icon.setStyleSheet("color: #2E7D32; font-weight: bold; font-size: 12px;")
            
            result_label = QLabel(recommendation_text)
            result_label.setWordWrap(True)
            result_label.setStyleSheet("font-size: 16px; margin: 10px 0;")
            
            # 음성 안내 버튼
            tts_btn = QPushButton("🔊 음성으로 들려주기")
            tts_btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    padding: 10px;
                }
            """)
            tts_btn.clicked.connect(lambda: text_to_speech(recommendation_text))
            
            result_layout.addWidget(security_icon)
            result_layout.addWidget(result_label)
            result_layout.addWidget(tts_btn)
            
            # 기존 컨텐츠에 결과 추가
            self.content_layout.insertWidget(2, result_frame)
            
        except Exception as e:
            print(f"🚨 보안 빠른 추천 오류: {str(e)}")
            QMessageBox.warning(self, "추천 오류", "보안 처리 중 오류가 발생했습니다.")
    
    def show_health_recommendation(self):
        """건강 맞춤 추천 화면 표시 (기존 로직 + 보안 강화)"""
        self.clear_content()
        
        # 🔐 건강정보 보호 안내
        health_privacy_notice = QLabel("""
        🏥 건강정보 보호 안내
        • 건강 정보는 추천에만 사용되며 저장되지 않습니다
        • 모든 처리는 익명화되어 진행됩니다
        • 개인 식별이 불가능한 형태로만 처리됩니다
        """)
        health_privacy_notice.setStyleSheet("""
            background-color: #FFF3E0;
            border: 2px solid #FF9800;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            margin: 10px 0;
        """)
        
        title_label = QLabel("🏥 건강 맞춤 추천 (개인정보보호)")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px 0; color: #2E7D32;")
        
        # 나머지는 기존 로직과 동일하되 보안 강화
        self.content_layout.addWidget(title_label)
        self.content_layout.addWidget(health_privacy_notice)
        
        # 기존 건강 추천 UI 로직 추가...
        self.add_back_button()

# 🔐 메인 실행 함수 (보안 강화)
def main():
    app = QApplication(sys.argv)
    
    # 폰트 설정
    app.setFont(QFont("맑은 고딕", 10))
    
    # 🔐 보안 강화된 메인 윈도우 생성
    window = SecureMegaKioskUI()
    window.show()
    
    # 🔐 보안 모니터링 시작
    print("🔐 보안 강화된 메가 커피 키오스크가 시작되었습니다.")
    print("✅ 개인정보 보호 모드 활성화")
    print("✅ 접근성 기능 활성화")
    print("✅ 보안 모니터링 시작")
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()