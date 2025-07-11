# identity_verification.py
import cv2
import numpy as np
import easyocr
import re
from datetime import datetime, date
import hashlib

class IdentityVerificationSystem:
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['ko', 'en'])
        self.min_age_for_caffeine = 14  # 카페인 제한 연령
        self.max_daily_caffeine = {
            'youth': 100,    # 청소년 (14-18세)
            'adult': 400,    # 성인 (19-64세) 
            'senior': 300    # 노인 (65세 이상)
        }
        
    def detect_id_card(self, image):
        """신분증 감지 및 영역 추출"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 신분증 크기 비율 검사 (가로:세로 = 약 1.6:1)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # 사각형 근사
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:  # 사각형
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h
                
                # 신분증 비율 확인 (1.4 ~ 1.8)
                if 1.4 <= aspect_ratio <= 1.8 and w > 200 and h > 100:
                    return image[y:y+h, x:x+w]
        
        return None
    
    def extract_birth_date(self, id_text):
        """주민등록번호에서 생년월일 추출"""
        # 주민등록번호 패턴 (6자리-7자리 또는 6자리-1자리***)
        patterns = [
            r'(\d{2})(\d{2})(\d{2})-([1-4])\d{6}',  # 전체 주민번호
            r'(\d{2})(\d{2})(\d{2})-([1-4])\*{6}',  # 뒷자리 마스킹
            r'(\d{6})-([1-4])',                      # 앞 6자리만
        ]
        
        for pattern in patterns:
            match = re.search(pattern, id_text)
            if match:
                if len(match.groups()) >= 4:
                    year, month, day, gender = match.groups()[:4]
                else:
                    year, month, day = match.group(1)[:2], match.group(1)[2:4], match.group(1)[4:6]
                    gender = match.group(2)
                
                # 년도 보정 (1900년대/2000년대 구분)
                if gender in ['1', '2']:
                    full_year = 1900 + int(year)
                elif gender in ['3', '4']:
                    full_year = 2000 + int(year)
                else:
                    continue
                
                try:
                    birth_date = date(full_year, int(month), int(day))
                    return birth_date
                except ValueError:
                    continue
        
        return None
    
    def calculate_age(self, birth_date):
        """나이 계산"""
        today = date.today()
        age = today.year - birth_date.year
        
        # 생일이 지나지 않았으면 -1
        if today.month < birth_date.month or \
           (today.month == birth_date.month and today.day < birth_date.day):
            age -= 1
            
        return age
    
    def verify_age_restriction(self, age, selected_menu, menu_df):
        """연령 제한 확인"""
        restrictions = []
        
        # 카페인 함량 확인
        menu_info = menu_df[menu_df['이름'] == selected_menu]
        if not menu_info.empty:
            caffeine_mg = menu_info.iloc[0].get('카페인(mg)', 0)
            
            # 14세 미만 카페인 제한
            if age < self.min_age_for_caffeine and caffeine_mg > 0:
                restrictions.append({
                    'type': 'age_restriction',
                    'message': f'만 {self.min_age_for_caffeine}세 미만은 카페인 음료를 주문할 수 없습니다.',
                    'alternative': '디카페인 음료를 추천드립니다.'
                })
            
            # 연령대별 권장 카페인 한도 안내
            elif age < 19 and caffeine_mg > self.max_daily_caffeine['youth']:
                restrictions.append({
                    'type': 'caffeine_warning',
                    'message': f'청소년 일일 권장 카페인량({self.max_daily_caffeine["youth"]}mg)을 초과합니다.',
                    'alternative': '카페인이 적은 음료를 권장합니다.'
                })
        
        return restrictions
    
    def process_id_verification(self, image, selected_menu, menu_df):
        """신분증 인증 전체 프로세스"""
        result = {
            'verified': False,
            'age': None,
            'restrictions': [],
            'message': ''
        }
        
        try:
            # 1. 신분증 영역 감지
            id_region = self.detect_id_card(image)
            if id_region is None:
                result['message'] = '신분증을 인식할 수 없습니다. 신분증을 카메라에 가까이 대어주세요.'
                return result
            
            # 2. OCR로 텍스트 추출
            ocr_results = self.ocr_reader.readtext(id_region)
            extracted_text = ' '.join([text[1] for text in ocr_results])
            
            # 3. 생년월일 추출
            birth_date = self.extract_birth_date(extracted_text)
            if birth_date is None:
                result['message'] = '주민등록번호를 인식할 수 없습니다. 신분증의 글씨가 명확히 보이도록 해주세요.'
                return result
            
            # 4. 나이 계산
            age = self.calculate_age(birth_date)
            result['age'] = age
            
            # 5. 연령 제한 확인
            restrictions = self.verify_age_restriction(age, selected_menu, menu_df)
            result['restrictions'] = restrictions
            
            # 6. 인증 결과
            if not restrictions or all(r['type'] != 'age_restriction' for r in restrictions):
                result['verified'] = True
                result['message'] = f'본인인증이 완료되었습니다. (만 {age}세)'
            else:
                result['verified'] = False
                result['message'] = restrictions[0]['message']
            
            # 7. 개인정보 로그 (해시화)
            self.log_verification(birth_date, age, selected_menu, result['verified'])
            
        except Exception as e:
            result['message'] = f'인증 처리 중 오류가 발생했습니다: {str(e)}'
            print(f"신분증 인증 오류: {e}")
        
        return result
    
    def log_verification(self, birth_date, age, menu, verified):
        """인증 로그 기록 (개인정보 해시화)"""
        # 생년월일을 해시화하여 개인정보 보호
        birth_hash = hashlib.sha256(birth_date.isoformat().encode()).hexdigest()[:16]
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'birth_hash': birth_hash,  # 해시화된 생년월일
            'age_group': self.get_age_group(age),  # 구체적 나이 대신 연령대
            'menu_category': self.get_menu_category(menu),
            'verification_result': verified,
            'session_id': hashlib.md5(f"{datetime.now()}{birth_hash}".encode()).hexdigest()[:12]
        }
        
        with open("verification_log.json", "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def get_age_group(self, age):
        """연령대 분류"""
        if age < 13:
            return "어린이"
        elif age < 20:
            return "청소년"
        elif age < 65:
            return "성인" 
        else:
            return "노인"
    
    def get_menu_category(self, menu_name):
        """메뉴 카테고리 분류"""
        if any(keyword in menu_name for keyword in ['커피', '아메리카노', '라떼', '에스프레소']):
            return "커피"
        elif any(keyword in menu_name for keyword in ['차', '티', '녹차', '홍차']):
            return "차/티"
        elif any(keyword in menu_name for keyword in ['스무디', '프라페']):
            return "음료"
        else:
            return "기타"

# UI 통합을 위한 다이얼로그
class IdentityVerificationDialog:
    def __init__(self, parent, menu_name, menu_df):
        self.parent = parent
        self.menu_name = menu_name
        self.menu_df = menu_df
        self.verification_system = IdentityVerificationSystem()
        
    def start_verification(self):
        """신분증 인증 시작"""
        print("📱 신분증을 카메라에 대어주세요...")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 신분증 가이드 표시
            cv2.rectangle(frame, (50, 100), (590, 380), (0, 255, 0), 2)
            cv2.putText(frame, "ID Card Area", (55, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press SPACE to capture, ESC to cancel", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Identity Verification", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 스페이스바로 캡처
                result = self.verification_system.process_id_verification(frame, self.menu_name, self.menu_df)
                cap.release()
                cv2.destroyAllWindows()
                return result
            elif key == 27:  # ESC로 취소
                cap.release()
                cv2.destroyAllWindows()
                return {'verified': False, 'message': '인증이 취소되었습니다.'}
        
        cap.release()
        cv2.destroyAllWindows()
        return {'verified': False, 'message': '카메라 오류가 발생했습니다.'}