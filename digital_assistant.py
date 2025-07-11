# digital_assistant.py
import sqlite3
import json
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class DigitalAssistantSystem:
    def __init__(self):
        self.db_path = "kiosk_assistance.db"
        self.init_database()
        self.difficulty_threshold = 3  # 3번 이상 도움 요청 시 고도화 필요
        
    def init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 도움 요청 로그 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assistance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                user_age_group TEXT,
                assistance_type TEXT,
                step_name TEXT,
                help_count INTEGER,
                resolution_time REAL,
                satisfaction_score INTEGER
            )
        ''')
        
        # 사용자 피드백 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                session_id TEXT,
                feedback_type TEXT,
                rating INTEGER,
                comment TEXT,
                improvement_suggestion TEXT
            )
        ''')
        
        # 시스템 성능 메트릭 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_users INTEGER,
                assistance_requests INTEGER,
                avg_completion_time REAL,
                error_count INTEGER,
                satisfaction_avg REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def detect_user_difficulty(self, session_id, current_step):
        """사용자 어려움 감지"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 현재 세션의 도움 요청 횟수 확인
        cursor.execute('''
            SELECT COUNT(*) FROM assistance_log 
            WHERE session_id = ? AND step_name = ?
        ''', (session_id, current_step))
        
        help_count = cursor.fetchone()[0]
        conn.close()
        
        difficulty_level = {
            0: "정상",
            1: "약간 어려움", 
            2: "어려움",
            3: "매우 어려움"
        }.get(min(help_count, 3), "매우 어려움")
        
        return {
            'help_count': help_count,
            'difficulty_level': difficulty_level,
            'needs_assistance': help_count >= 2
        }
    
    def provide_contextual_help(self, step_name, user_age_group, difficulty_level):
        """상황별 맞춤 도움말 제공"""
        help_content = {
            "메뉴_선택": {
                "어린이": {
                    "text": "어떤 음료를 마시고 싶나요? 화면의 그림을 터치해보세요!",
                    "voice": True,
                    "visual_guide": "highlight_menu_images"
                },
                "노인": {
                    "text": "원하시는 메뉴를 천천히 선택해주세요. 글씨가 작다면 확대해드릴까요?",
                    "voice": True,
                    "font_size_increase": True
                }
            },
            "온도_선택": {
                "공통": {
                    "text": "따뜻한 음료는 HOT, 차가운 음료는 ICE를 선택해주세요.",
                    "voice": True,
                    "visual_guide": "highlight_temperature_buttons"
                }
            },
            "결제": {
                "노인": {
                    "text": "카드를 아래쪽 카드 리더기에 넣어주세요. 삐- 소리가 날 때까지 기다려주세요.",
                    "voice": True,
                    "animation": "card_insertion_guide"
                },
                "공통": {
                    "text": "결제 방법을 선택해주세요. 카드 또는 현금이 가능합니다.",
                    "voice": True
                }
            }
        }
        
        # 연령대별 또는 공통 도움말 선택
        step_help = help_content.get(step_name, {})
        user_help = step_help.get(user_age_group, step_help.get("공통", {}))
        
        # 어려움 정도에 따른 추가 지원
        if difficulty_level in ["매우 어려움", "어려움"]:
            user_help["call_staff"] = True
            user_help["extended_time"] = True
            
        return user_help
    
    def log_assistance_request(self, session_id, user_age_group, assistance_type, step_name):
        """도움 요청 로깅"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assistance_log 
            (timestamp, session_id, user_age_group, assistance_type, step_name, help_count)
            VALUES (?, ?, ?, ?, ?, 1)
        ''', (
            datetime.now().isoformat(),
            session_id,
            user_age_group,
            assistance_type,
            step_name
        ))
        
        conn.commit()
        conn.close()
    
    def collect_user_feedback(self, session_id, feedback_data):
        """사용자 피드백 수집"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_feedback 
            (timestamp, session_id, feedback_type, rating, comment, improvement_suggestion)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            session_id,
            feedback_data.get('type', 'general'),
            feedback_data.get('rating', 0),
            feedback_data.get('comment', ''),
            feedback_data.get('suggestion', '')
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_usage_patterns(self):
        """사용 패턴 분석 및 개선점 도출"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 최근 7일간 데이터 분석
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        
        # 가장 어려워하는 단계 분석
        cursor.execute('''
            SELECT step_name, COUNT(*) as help_requests, user_age_group
            FROM assistance_log 
            WHERE timestamp > ?
            GROUP BY step_name, user_age_group
            ORDER BY help_requests DESC
        ''', (week_ago,))
        
        difficulty_analysis = cursor.fetchall()
        
        # 연령대별 사용 패턴
        cursor.execute('''
            SELECT user_age_group, 
                   COUNT(*) as total_requests,
                   AVG(help_count) as avg_help_per_session
            FROM assistance_log 
            WHERE timestamp > ?
            GROUP BY user_age_group
        ''', (week_ago,))
        
        age_patterns = cursor.fetchall()
        
        # 만족도 분석
        cursor.execute('''
            SELECT AVG(rating) as avg_satisfaction,
                   COUNT(*) as feedback_count
            FROM user_feedback 
            WHERE timestamp > ?
        ''', (week_ago,))
        
        satisfaction_data = cursor.fetchone()
        
        conn.close()
        
        return {
            'difficult_steps': difficulty_analysis,
            'age_patterns': age_patterns,
            'satisfaction': {
                'average': satisfaction_data[0] if satisfaction_data[0] else 0,
                'response_count': satisfaction_data[1]
            }
        }
    
    def generate_improvement_recommendations(self):
        """개선 권고사항 생성"""
        analysis = self.analyze_usage_patterns()
        recommendations = []
        
        # 어려운 단계 개선
        for step, count, age_group in analysis['difficult_steps'][:3]:
            if count > 10:  # 주간 10회 이상 도움 요청
                recommendations.append({
                    'priority': 'high',
                    'area': step,
                    'target_group': age_group,
                    'suggestion': f'{age_group} 사용자를 위한 {step} 단계 UI 개선 필요',
                    'action': f'{step} 화면의 버튼 크기 확대 및 안내 메시지 추가'
                })
        
        # 만족도 기반 권고
        if analysis['satisfaction']['average'] < 3.5:
            recommendations.append({
                'priority': 'high',
                'area': '전체_UX',
                'suggestion': '전반적인 사용자 만족도 개선 필요',
                'action': '사용자 인터뷰 및 UI/UX 재설계 검토'
            })
        
        return recommendations
    
    def send_daily_report(self, admin_email):
        """일일 리포트 발송"""
        analysis = self.analyze_usage_patterns()
        recommendations = self.generate_improvement_recommendations()
        
        report_content = f"""
        📊 메가 커피 키오스크 일일 리포트
        
        📈 사용 현황:
        - 총 피드백 수: {analysis['satisfaction']['response_count']}개
        - 평균 만족도: {analysis['satisfaction']['average']:.1f}/5.0
        
        🚨 주요 이슈:
        """
        
        for rec in recommendations[:3]:
            report_content += f"- {rec['suggestion']}\n"
        
        report_content += f"""
        
        📋 상세 분석:
        어려운 단계 TOP 3:
        """
        
        for step, count, age_group in analysis['difficult_steps'][:3]:
            report_content += f"- {step} ({age_group}): {count}회 도움 요청\n"
        
        # 이메일 발송 (실제 환경에서는 SMTP 설정 필요)
        try:
            self.send_email(admin_email, "키오스크 일일 리포트", report_content)
        except Exception as e:
            print(f"리포트 발송 실패: {e}")
    
    def send_email(self, to_email, subject, content):
        """이메일 발송 함수"""
        # 실제 운영시에는 SMTP 서버 정보 설정 필요
        print(f"📧 리포트 발송 대상: {to_email}")
        print(f"제목: {subject}")
        print(content)

class AccessibilityEnhancer:
    """접근성 향상 기능"""
    
    def __init__(self):
        self.font_scale = 1.0
        self.high_contrast = False
        self.voice_guidance = True
        
    def adjust_for_visual_impairment(self, severity="medium"):
        """시각 장애 대응 조정"""
        adjustments = {
            "mild": {
                "font_scale": 1.2,
                "button_size_multiplier": 1.1,
                "contrast_boost": 1.1
            },
            "medium": {
                "font_scale": 1.5,
                "button_size_multiplier": 1.3,
                "contrast_boost": 1.3,
                "enable_voice": True
            },
            "severe": {
                "font_scale": 2.0,
                "button_size_multiplier": 1.5,
                "high_contrast_mode": True,
                "enable_voice": True,
                "screen_reader_compatible": True
            }
        }
        
        return adjustments.get(severity, adjustments["medium"])
    
    def provide_voice_navigation(self, current_screen, available_options):
        """음성 네비게이션 제공"""
        navigation_text = f"현재 {current_screen} 화면입니다. "
        
        if available_options:
            navigation_text += "선택 가능한 옵션은 다음과 같습니다: "
            navigation_text += ", ".join(available_options)
            navigation_text += ". 원하는 옵션을 말씀해주세요."
        
        return navigation_text
    
    def enable_gesture_control(self):
        """제스처 컨트롤 활성화"""
        gesture_commands = {
            "swipe_left": "이전 메뉴",
            "swipe_right": "다음 메뉴", 
            "tap": "선택",
            "double_tap": "확인",
            "long_press": "도움말"
        }
        
        return gesture_commands

# 통합 접근성 시스템
class IntegratedAccessibilitySystem:
    def __init__(self, main_ui):
        self.main_ui = main_ui
        self.assistant = DigitalAssistantSystem()
        self.enhancer = AccessibilityEnhancer()
        self.current_session = self.generate_session_id()
        
    def generate_session_id(self):
        """세션 ID 생성"""
        import uuid
        return str(uuid.uuid4())[:12]
    
    def start_accessibility_mode(self, user_profile):
        """접근성 모드 시작"""
        # 사용자 프로필에 따른 설정 조정
        if user_profile.get('age_group') == '노인':
            adjustments = self.enhancer.adjust_for_visual_impairment("medium")
            self.apply_ui_adjustments(adjustments)
        
        if user_profile.get('disabilities'):
            for disability in user_profile['disabilities']:
                if 'visual' in disability.lower():
                    self.enable_screen_reader_mode()
                elif 'motor' in disability.lower():
                    self.enable_large_button_mode()
    
    def apply_ui_adjustments(self, adjustments):
        """UI 조정사항 적용"""
        # PyQt5 UI에 조정사항 적용
        if hasattr(self.main_ui, 'setStyleSheet'):
            style_updates = f"""
            QLabel {{ font-size: {int(14 * adjustments.get('font_scale', 1.0))}px; }}
            QPushButton {{ 
                font-size: {int(16 * adjustments.get('font_scale', 1.0))}px;
                min-height: {int(50 * adjustments.get('button_size_multiplier', 1.0))}px;
            }}
            """
            if adjustments.get('high_contrast_mode'):
                style_updates += """
                QWidget { background-color: #000000; color: #FFFFFF; }
                QPushButton { background-color: #FFFFFF; color: #000000; }
                """
            
            self.main_ui.setStyleSheet(style_updates)
    
    def request_assistance(self, step_name, assistance_type="general"):
        """도움 요청 처리"""
        user_age_group = getattr(self.main_ui, 'current_user_age_group', '성인')
        
        # 어려움 정도 감지
        difficulty = self.assistant.detect_user_difficulty(self.current_session, step_name)
        
        # 맞춤 도움말 제공
        help_content = self.assistant.provide_contextual_help(
            step_name, user_age_group, difficulty['difficulty_level']
        )
        
        # 도움 요청 로깅
        self.assistant.log_assistance_request(
            self.current_session, user_age_group, assistance_type, step_name
        )
        
        # 스태프 호출이 필요한 경우
        if help_content.get('call_staff'):
            self.call_staff_assistance()
        
        return help_content
    
    def call_staff_assistance(self):
        """스태프 도움 호출"""
        print("🔔 스태프 호출이 요청되었습니다.")
        # 실제 환경에서는 스태프 호출 시스템 연동
        # 예: 카운터 알림, 관리자 앱 푸시 알림 등
        
    def collect_feedback(self, rating, comment="", suggestion=""):
        """피드백 수집"""
        feedback_data = {
            'type': 'usage_experience',
            'rating': rating,
            'comment': comment,
            'suggestion': suggestion
        }
        
        self.assistant.collect_user_feedback(self.current_session, feedback_data)
        
        # 낮은 평점인 경우 즉시 알림
        if rating <= 2:
            print(f"⚠️ 낮은 만족도 감지: {rating}/5 - 즉시 개선 필요")
            
    def end_session(self):
        """세션 종료 처리"""
        # 새 세션 ID 생성
        self.current_session = self.generate_session_id()
        print("✅ 접근성 세션이 종료되었습니다.")