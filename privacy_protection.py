# privacy_protection.py (수정된 버전)
import hashlib
import hmac
import base64
import cv2
import numpy as np
from cryptography.fernet import Fernet
from datetime import datetime, timedelta
import os

class PrivacyProtectionManager:
    def __init__(self):
        self.encryption_key = self.generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        self.face_data_ttl = 300  # 5분 후 자동 삭제
        
    def generate_encryption_key(self):
        """암호화 키 생성 (키 관리 시스템 연동 필요)"""
        key_file = "encryption.key"
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def mask_face_features(self, face_image):
        """얼굴 특징점 마스킹"""
        try:
             # 🔧 이미지 타입 안전 처리
            if face_image.dtype != np.uint8:
                face_image = face_image.astype(np.uint8)
            # 얼굴 이미지를 해시값으로 변환
            face_hash = hashlib.sha256(face_image.tobytes()).hexdigest()
            
            # 원본 이미지는 즉시 삭제하고 해시만 보관
            masked_face = {
                'face_id': face_hash[:16],  # 첫 16자리만 사용
                'timestamp': datetime.now(),
                #'features': self.extract_minimal_features(face_image)
                'features': {'safe_mode': True}  # 간단한 특징으로 대체
            }
        
            return masked_face
        except Exception as e:
            print(f"⚠️ 마스킹 처리 실패: {e}")
            return {
                'face_id': 'anonymous',
                'timestamp': datetime.now(),
                'features': {'safe_mode': True}
            }
    
    def extract_minimal_features(self, face_image):
        """최소한의 특징만 추출 (연령 추정용)"""
        # 얼굴 특징을 수치화하되 개인 식별 불가능하게 처리
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # 간단한 텍스처 분석 (개인 식별 불가)
        features = {
            'texture_variance': np.var(gray),
            'brightness_avg': np.mean(gray),
            'estimated_age_range': None  # 추후 연령 범위만 저장
        }
        
        return features
    
    def anonymize_health_data(self, health_info):
        """건강정보 익명화"""
        if not health_info:
            return None
            
        # 질병명을 카테고리로 그룹화
        disease_categories = {
            '당뇨병': 'metabolic',
            '고혈압': 'cardiovascular', 
            '심장질환': 'cardiovascular',
            '고지혈증': 'metabolic'
        }
        
        anonymized = {
            'health_category': [disease_categories.get(disease, 'other') 
                              for disease in health_info],
            'risk_level': self.calculate_risk_level(health_info),
            'session_id': self.generate_session_id()
        }
        
        return anonymized
    
    def calculate_risk_level(self, diseases):
        """위험도 수준 계산 (구체적 질병명 노출 없이)"""
        if not diseases:
            return 'low'
        elif len(diseases) >= 2:
            return 'high'
        else:
            return 'medium'
    
    def generate_session_id(self):
        """세션별 임시 ID 생성"""
        return hashlib.md5(
            f"{datetime.now()}{os.urandom(16)}".encode()
        ).hexdigest()[:12]
    
    def encrypt_sensitive_data(self, data):
        """민감데이터 암호화"""
        json_data = str(data).encode()
        encrypted_data = self.fernet.encrypt(json_data)
        return base64.b64encode(encrypted_data).decode()
    
    def auto_delete_expired_data(self):
        """만료된 데이터 자동 삭제"""
        current_time = datetime.now()
        
        # 임시 파일 정리
        temp_files = ['speech_*.mp3', '*.tmp', 'face_temp_*']
        for pattern in temp_files:
            import glob
            for file in glob.glob(pattern):
                try:
                    file_time = datetime.fromtimestamp(os.path.getctime(file))
                    if (current_time - file_time).seconds > self.face_data_ttl:
                        os.remove(file)
                        print(f"🗑️ 만료된 임시파일 삭제: {file}")
                except:
                    pass

# ⚠️ 기존 FaceRecognitionKiosk 클래스 상속 제거
# 대신 독립적인 보안 얼굴 인식 클래스 생성
class SecureFaceAnalyzer:
    """보안 강화된 얼굴 분석 시스템"""
    
    def __init__(self):
        self.privacy_manager = PrivacyProtectionManager()
        
    def analyze_face_secure(self, face_crop_resized):
        """개인정보 보호 강화된 얼굴 분석"""
        try:
            # ⚠️ DeepFace import를 함수 내부로 이동
            try:
                from deepface import DeepFace
            except ImportError:
                print("❌ DeepFace가 설치되지 않음. pip install deepface")
                return False
            
            # 얼굴 데이터 마스킹
            masked_face = self.privacy_manager.mask_face_features(face_crop_resized)
            
            # 연령만 추출 (개인 식별정보 제거)
            analysis = DeepFace.analyze(face_crop_resized, actions=["age"], enforce_detection=False)
            age = int(analysis[0]["age"])
            
            # 연령대로 변환 (구체적 나이 저장 안함)
            age_group = self.get_age_group(age)
            
            # 익명화된 로그만 저장
            anonymous_log = {
                'session_id': masked_face['face_id'],
                'age_group': age_group,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # 원본 이미지 즉시 메모리에서 제거
            del face_crop_resized
            
            self.save_anonymous_log(anonymous_log)
            print(f"[보안 분석 완료] 연령대: {age_group}")
            
            return age_group  # 연령대 반환
            
        except Exception as e:
            print("[오류] 보안 얼굴 분석 실패:", e)
            return None
    
    def get_age_group(self, age):
        """연령을 그룹으로 변환"""
        if age < 13:
            return "어린이"
        elif age < 20:
            return "청소년" 
        elif age < 40:
            return "성인"
        else:
            return "노인"
    
    def save_anonymous_log(self, log_data):
        """익명화된 로그 저장"""
        encrypted_log = self.privacy_manager.encrypt_sensitive_data(log_data)
        
        with open("anonymous_usage.log", "a", encoding="utf-8") as f:
            f.write(f"{encrypted_log}\n")
            
    def cleanup_session(self):
        """세션 종료 시 데이터 정리"""
        self.privacy_manager.auto_delete_expired_data()
        print("🧹 세션 데이터 정리 완료")