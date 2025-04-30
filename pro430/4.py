import cv2
import os
import time
import csv
from datetime import datetime
from deepface import DeepFace
import pandas as pd
import numpy as np

# ---------------------- 설정 ----------------------
FRAME_BOX = ((150, 50), (500, 440))
CASCADE_PATH = 'data/haarcascade_frontalface_default.xml'
RESULT_CSV = '결과_로그.csv'
MENU_FILE = 'magacoffe.csv'

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------------------- 결과 로그 초기화 ----------------------
if not os.path.exists(RESULT_CSV):
    with open(RESULT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Age", "Recommended Menu"])

# ---------------------- 메뉴 데이터 로드 ----------------------
def load_menu_data_by_age(file_path=MENU_FILE):
    df = pd.read_csv(file_path)
    menu_by_age = {"청소년": [], "성인": [], "노인": []}
    for _, row in df.iterrows():
        age_text = row.get("연령", "")
        menu = row.get("메뉴", "")
        for category in menu_by_age.keys():
            if category in age_text:
                menu_by_age[category].append(menu)
    return menu_by_age

menu_by_age_dict = load_menu_data_by_age()

# ---------------------- 메뉴 추천 ----------------------
def get_menu_by_age(age):
    if age < 18:
        age_group = "청소년"
    elif age < 40:
        age_group = "성인"
    else:
        age_group = "노인"
    menus = menu_by_age_dict.get(age_group, [])
    if not menus:
        return "추천 메뉴가 없습니다"
    return f"{age_group} 추천 메뉴: {', '.join(menus[:3])} 등 총 {len(menus)}개"

# ---------------------- 유틸 함수 ----------------------
def resize_face(face_image, size=(224, 224)):
    return cv2.resize(face_image, size)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_different_person(ref_vector, cur_vector, ref_box, cur_box, analyzed):
    if ref_vector is None or cur_vector is None:
        return False
    # 벡터 유사도
    cosine = cosine_distance(ref_vector, cur_vector)

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

# ---------------------- 변수 초기화 ----------------------
cap = cv2.VideoCapture(0)
start_time = None
reference_vector = None
reference_box = None
analyzed = False
face_lost_time = None
GRACE_PERIOD = 2.0
printed_reset_message = False

# ---------------------- 메인 루프 ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (x1, y1), (x2, y2) = FRAME_BOX
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_in_box = False
    current_vector = None
    current_box = None
    largest_face = None
    max_area = 0

    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        area = w * h
        if x1 < center[0] < x2 and y1 < center[1] < y2 and area > max_area:
            largest_face = (x, y, w, h)
            max_area = area

    if largest_face is not None:
        x, y, w, h = largest_face
        current_box = (x, y, w, h)
        face_crop = frame[y:y+h, x:x+w]
        face_crop_resized = resize_face(face_crop)
        face_in_box = True

        try:
            result = DeepFace.represent(face_crop_resized, model_name="VGG-Face", enforce_detection=False)
            current_vector = result[0]["embedding"]
        except Exception as e:
            print("[오류] 얼굴 임베딩 실패:", e)
            continue

        if reference_vector is None:
            reference_vector = current_vector
            reference_box = current_box
            start_time = time.time()
            print("[저장됨] 기준 얼굴 임베딩 및 위치 저장 완료")
        else:
            if is_different_person(reference_vector, current_vector, reference_box, current_box, analyzed):
                print("[변경됨] 다른 사람 감지됨 → 분석 초기화")
                reference_vector = current_vector
                reference_box = current_box
                start_time = time.time()
                analyzed = False

        # 분석 실행
        elapsed = time.time() - start_time
        if elapsed >= 7 and not analyzed:
            try:
                analysis = DeepFace.analyze(face_crop_resized, actions=["age"], enforce_detection=False)
                age = int(analysis[0]["age"])
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                menu = get_menu_by_age(age)
                print(f"[분석 완료] 나이: {age}세 → {menu}")

                with open(RESULT_CSV, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, age, menu])

                analyzed = True
            except Exception as e:
                print("[오류] 나이 분석 실패:", e)

    # ✅ 유예 시간 기반 초기화
    if not face_in_box:
        if face_lost_time is None:
            face_lost_time = time.time()
        elif time.time() - face_lost_time > GRACE_PERIOD:
            if not printed_reset_message:
                print("[초기화] 얼굴 사라짐 - 새 사용자 대기 중")
                printed_reset_message = True

            reference_vector = None
            reference_box = None
            analyzed = False
            start_time = None
    else:
        face_lost_time = None
        printed_reset_message = False

    cv2.imshow("Kiosk", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
