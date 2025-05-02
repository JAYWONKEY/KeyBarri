import cv2
import os
import time
import csv
from datetime import datetime
from deepface import DeepFace
import pandas as pd
import numpy as np

class FaceRecognitionKiosk:
    def __init__(self):
        # ---------------------- 설정 ----------------------
        self.FRAME_BOX = ((150, 50), (500, 440))
        self.CASCADE_PATH = 'data/haarcascade_frontalface_default.xml'
        self.RESULT_CSV = '결과_로그.csv'
        self.MENU_FILE = '123/process_data.csv'
        self.GRACE_PERIOD = 2.0
        
        # 초기화
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)
        self.cap = cv2.VideoCapture(0)
        self.start_time = None
        self.reference_vector = None
        self.reference_box = None
        self.analyzed = False
        self.face_lost_time = None
        self.printed_reset_message = False
        
        # 메뉴 데이터 로드
        self.menu_by_age_dict = self.load_menu_data_by_age()
        
        # 결과 로그 초기화
        self._initialize_result_log()
    
    def _initialize_result_log(self):
        """결과 로그 CSV 파일 초기화"""
        if not os.path.exists(self.RESULT_CSV):
            with open(self.RESULT_CSV, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Age", "Recommended Menu"])
    
    def load_menu_data_by_age(self, file_path=None):
        """연령대별 메뉴 데이터 로드"""
        if file_path is None:
            file_path = self.MENU_FILE
            
        df = pd.read_csv(file_path)
        menu_by_age = {"청소년": [], "성인": [], "노인": []}
        for _, row in df.iterrows():
            age_text = row.get("연령", "")
            menu = row.get("메뉴", "")
            for category in menu_by_age.keys():
                if category in age_text:
                    menu_by_age[category].append(menu)
        return menu_by_age
    
    def get_menu_by_age(self, age):
        """나이에 따른 메뉴 추천"""
        if age < 18:
            age_group = "청소년"
        elif age < 40:
            age_group = "성인"
        else:
            age_group = "노인"
        menus = self.menu_by_age_dict.get(age_group, [])
        if not menus:
            return "추천 메뉴가 없습니다"
        return f"{age_group} 추천 메뉴: {', '.join(menus[:3])} 등 총 {len(menus)}개"
    
    @staticmethod
    def resize_face(face_image, size=(224, 224)):
        """얼굴 이미지 리사이즈"""
        return cv2.resize(face_image, size)
    
    @staticmethod
    def cosine_distance(a, b):
        """벡터 간 코사인 거리 계산"""
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def is_different_person(self, ref_vector, cur_vector, ref_box, cur_box, analyzed):
        """동일인 여부 판별"""
        if ref_vector is None or cur_vector is None:
            return False
        # 벡터 유사도
        cosine = self.cosine_distance(ref_vector, cur_vector)

        # 중심 거리
        x1, y1, w1, h1 = ref_box
        x2, y2, w2, h2 = cur_box
        center_dist = np.linalg.norm([(x1 + w1 / 2) - (x2 + w2 / 2),
                                      (y1 + h1 / 2) - (y2 + h2 / 2)])
        # 크기 변화 비율
        area1 = w1 * h1
        area2 = w2 * h2
        size_ratio = abs(area1 - area2) / area1 if area1 > 0 else 0

        # ✅ 분석 전일 때만 판별
        if not analyzed and cosine > 0.45 and (center_dist > 100 or size_ratio > 0.3):
            print(f"[유사도 거리: {cosine:.3f}] 중심이동: {center_dist:.1f}, 크기변화: {size_ratio:.2f}")
            return True
        return False
    
    def analyze_face(self, face_crop_resized):
        """얼굴 분석 및 결과 기록"""
        try:
            analysis = DeepFace.analyze(face_crop_resized, actions=["age"], enforce_detection=False)
            age = int(analysis[0]["age"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            menu = self.get_menu_by_age(age)
            print(f"[분석 완료] 나이: {age}세 → {menu}")

            with open(self.RESULT_CSV, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, age, menu])

            return True
        except Exception as e:
            print("[오류] 나이 분석 실패:", e)
            return False
    
    def process_frame(self):
        """프레임 처리 및 얼굴 인식"""
        ret, frame = self.cap.read()
        if not ret:
            return False
        
        (x1, y1), (x2, y2) = self.FRAME_BOX
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        face_in_box = False
        current_vector = None
        current_box = None
        largest_face = None
        max_area = 0

        # 가장 큰 얼굴 찾기
        for (x, y, w, h) in faces:
            center = (x + w // 2, y + h // 2)
            area = w * h
            if x1 < center[0] < x2 and y1 < center[1] < y2 and area > max_area:
                largest_face = (x, y, w, h)
                max_area = area

        # 얼굴이 감지된 경우 처리
        if largest_face is not None:
            x, y, w, h = largest_face
            current_box = (x, y, w, h)
            face_crop = frame[y:y+h, x:x+w]
            face_crop_resized = self.resize_face(face_crop)
            face_in_box = True

            try:
                result = DeepFace.represent(face_crop_resized, model_name="VGG-Face", enforce_detection=False)
                current_vector = result[0]["embedding"]
            except Exception as e:
                print("[오류] 얼굴 임베딩 실패:", e)
                cv2.imshow("Kiosk", frame)
                return True

            # 기준 얼굴이 없으면 새로 설정
            if self.reference_vector is None:
                self.reference_vector = current_vector
                self.reference_box = current_box
                self.start_time = time.time()
                print("[저장됨] 기준 얼굴 임베딩 및 위치 저장 완료")
            else:
                # 다른 사람 감지 시 초기화
                if self.is_different_person(self.reference_vector, current_vector, 
                                           self.reference_box, current_box, self.analyzed):
                    print("[변경됨] 다른 사람 감지됨 → 분석 초기화")
                    self.reference_vector = current_vector
                    self.reference_box = current_box
                    self.start_time = time.time()
                    self.analyzed = False

            # 분석 실행 (7초 대기 후)
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                if elapsed >= 7 and not self.analyzed:
                    self.analyzed = self.analyze_face(face_crop_resized)

        # 얼굴이 범위를 벗어난 경우 처리
        self._handle_face_absence(face_in_box)
        
        cv2.imshow("Kiosk", frame)
        return True
    
    def _handle_face_absence(self, face_in_box):
        """얼굴이 없을 때 처리 로직"""
        if not face_in_box:
            if self.face_lost_time is None:
                self.face_lost_time = time.time()
            elif time.time() - self.face_lost_time > self.GRACE_PERIOD:
                if not self.printed_reset_message:
                    print("[초기화] 얼굴 사라짐 - 새 사용자 대기 중")
                    self.printed_reset_message = True

                self.reference_vector = None
                self.reference_box = None
                self.analyzed = False
                self.start_time = None
        else:
            self.face_lost_time = None
            self.printed_reset_message = False
    
    
    def run(self):
        """메인 루프 실행"""
        while True:
            if not self.process_frame():
                break
                
            if cv2.waitKey(1) == 27:  # ESC 키
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

# 메인 실행 부분


def get_age_from_camera():
    """카메라로부터 얼굴을 캡처하고 나이를 추정하여 반환"""
    import cv2
    from deepface import DeepFace

    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    age = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_crop_resized = cv2.resize(face_crop, (224, 224))
            try:
                analysis = DeepFace.analyze(face_crop_resized, actions=["age"], enforce_detection=False)
                age = int(analysis[0]["age"])
                break
            except:
                continue

        if age:
            break

        cv2.imshow("Face Capture", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return age

if __name__ == "__main__":
    kiosk = FaceRecognitionKiosk()
    kiosk.run()