import cv2
from deepface import DeepFace
import json
from datetime import datetime
import time
import pandas as pd
from utils import put_korean_text

class FaceAnalyzer:
    def __init__(self, cascade_path=None, menu_file="data.csv"):
        """초기화"""
        # Cascade 파일 경로 설정
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Cascade Classifier 로드 실패")
            
        self.cap = None
        self.last_detected_face = None
        self.detection_start_time = None
        self.age_stored = False
        
        # 메뉴 데이터 로드
        try:
            self.menu_df = pd.read_csv(menu_file, encoding='cp949')
            self.menu_recommendations = {}
            
            # 연령대별 메뉴 그룹화
            age_groups = {
                '어린이': self.menu_df[self.menu_df['연령대'] == '어린이']['메뉴명'].tolist(),
                '청소년': self.menu_df[self.menu_df['연령대'] == '청소년']['메뉴명'].tolist(),
                '성인': self.menu_df[self.menu_df['연령대'] == '성인']['메뉴명'].tolist(),
                '노인': self.menu_df[self.menu_df['연령대'] == '노인']['메뉴명'].tolist()
            }
            self.menu_recommendations = age_groups
            print("메뉴 데이터 로드 완료")
            
        except Exception as e:
            print(f"메뉴 데이터 로드 실패: {str(e)}")
            # 기본 메뉴 설정
            self.menu_recommendations = {
                '어린이': ['초코라떼', '바닐라라떼', '캐러멜라떼', '딸기스무디', '초코스무디'],
                '청소년': ['아메리카노', '카페라떼', '카페모카', '캐러멜마끼아또', '프라푸치노'],
                '성인': ['아메리카노', '에스프레소', '카페라떼', '바닐라라떼', '콜드브루'],
                '노인': ['아메리카노', '카페라떼', '디카페인커피', '녹차라떼', '얼그레이티']
            }

    def get_age_group(self, age):
        """연령대 결정"""
        if age < 13: return '어린이'
        elif age < 20: return '청소년'
        elif age < 60: return '성인'
        else: return '노인'

    def start_camera(self):
        """카메라 시작"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("카메라를 열 수 없습니다")

    def detect_and_analyze(self, frame):
        """얼굴 감지 및 분석"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_roi = frame[y:y+h, x:x+w]
                
                try:
                    result = DeepFace.analyze(face_roi, actions=['age'], enforce_detection=False)
                    age = int(result[0]['age'])
                    age_group = self.get_age_group(age)
                    recommendations = self.menu_recommendations[age_group]

                    # 결과 표시
                    frame = put_korean_text(frame, f'나이: {age}세', (x, y-60))
                    frame = put_korean_text(frame, f'연령대: {age_group}', (x, y-30))

                    # 결과 저장
                    self.save_results({
                        '시간': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        '나이': age,
                        '연령대': age_group,
                        '추천메뉴': recommendations
                    })

                except Exception as e:
                    print(f"얼굴 분석 오류: {e}")

            return frame

        except Exception as e:
            print(f"프레임 처리 오류: {e}")
            return frame

    def save_results(self, results):
        """결과 저장"""
        try:
            with open("analysis_results.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"결과 저장 오류: {e}")

    def run(self):
        """메인 실행"""
        if self.cap is None:
            self.start_camera()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = self.detect_and_analyze(frame)
                cv2.imshow('Face Analysis', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()